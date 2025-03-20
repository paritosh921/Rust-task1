use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use rayon::prelude::*;
use tch::{nn, Tensor};
use clap::{App, Arg};
use serde::{Deserialize, Serialize};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    input_dir: String,
    output_dir: String,
    model_path: String,
    min_confidence: f32,
    target_size: u32,
    padding_factor: f32,
    max_faces: usize,
    batch_size: usize,
}

#[derive(Debug, Clone)]
struct Detection {
    bbox: [f32; 4],
    confidence: f32,
}

struct FaceDetector {
    model: nn::Module,
    device: tch::Device,
}

impl FaceDetector {
    fn new(model_path: &str) -> Result<Self> {
        let device = tch::Device::cuda_if_available();
        println!("Using device: {:?}", device);
        
        let model = tch::CModule::load(model_path)
            .with_context(|| format!("Failed to load model from {}", model_path))?;
            
        Ok(Self {
            model: model.into(),
            device,
        })
    }
    
    fn detect(&self, batch: &Tensor) -> Result<Vec<Vec<Detection>>> {
        let batch = batch.to(self.device);
        
        let output = self.model.forward_ts(&[batch])?;
        
        let batch_size = batch.size()[0] as usize;
        let output = output.detach();
        
        let mut all_detections = Vec::with_capacity(batch_size);
        
        for i in 0..batch_size {
            let detections = self.process_predictions(&output.select(0, i as i64))?;
            all_detections.push(detections);
        }
        
        Ok(all_detections)
    }
    
    fn process_predictions(&self, predictions: &Tensor) -> Result<Vec<Detection>> {
        let predictions = predictions.to_device(tch::Device::Cpu);
        let predictions_view = predictions.view([-1, 5]);
        
        let mut detections = Vec::new();
        
        let num_preds = predictions_view.size()[0];
        
        let preds_accessor = predictions_view.float_value_2d();
        for i in 0..num_preds {
            let confidence = preds_accessor[i as usize][4];
            if confidence > 0.25 {
                let bbox = [
                    preds_accessor[i as usize][0],
                    preds_accessor[i as usize][1],
                    preds_accessor[i as usize][2],
                    preds_accessor[i as usize][3],
                ];
                
                detections.push(Detection { bbox, confidence });
            }
        }
        
        let mut filtered_detections = Vec::new();
        
        let mut indexed_detections: Vec<_> = detections.iter().enumerate().collect();
        indexed_detections.sort_by(|(_, a), (_, b)| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut is_suppressed = vec![false; indexed_detections.len()];
        
        for (i, (idx_i, det_i)) in indexed_detections.iter().enumerate() {
            if is_suppressed[i] {
                continue;
            }
            
            filtered_detections.push((*det_i).clone());
            
            for (j, (idx_j, det_j)) in indexed_detections.iter().enumerate() {
                if i == j || is_suppressed[j] {
                    continue;
                }
                
                let iou = compute_iou(&det_i.bbox, &det_j.bbox);
                if iou > 0.45 {
                    is_suppressed[j] = true;
                }
            }
        }
        
        Ok(filtered_detections)
    }
}

fn compute_iou(bbox1: &[f32; 4], bbox2: &[f32; 4]) -> f32 {
    let x1 = bbox1[0].max(bbox2[0]);
    let y1 = bbox1[1].max(bbox2[1]);
    let x2 = bbox1[2].min(bbox2[2]);
    let y2 = bbox1[3].min(bbox2[3]);
    
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    let area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    let union = area1 + area2 - intersection;
    
    if union <= 0.0 {
        return 0.0;
    }
    
    intersection / union
}

fn preprocess_images(images: &[DynamicImage], target_size: u32) -> Result<Tensor> {
    let batch_size = images.len();
    
    let mut batch_tensor = Tensor::zeros(&[batch_size as i64, 3, target_size as i64, target_size as i64], 
                                         tch::kind::FLOAT_CPU);
    
    for (i, img) in images.iter().enumerate() {
        let resized = img.resize_exact(target_size, target_size, 
                                      image::imageops::FilterType::Lanczos3);
        
        let tensor = image_to_tensor(&resized)?;
        
        batch_tensor.select(0, i as i64).copy_(&tensor);
    }
    
    let mean = Tensor::from_slice(&[0.485, 0.456, 0.406]);
    let std = Tensor::from_slice(&[0.229, 0.224, 0.225]);
    
    let normalized = batch_tensor
        .permute(&[0, 2, 3, 1])
        .sub(mean)
        .div(std)
        .permute(&[0, 3, 1, 2]);
    
    Ok(normalized)
}

fn image_to_tensor(img: &DynamicImage) -> Result<Tensor> {
    let (width, height) = img.dimensions();
    let mut tensor = Tensor::zeros(&[3, height as i64, width as i64], tch::kind::FLOAT_CPU);
    
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            tensor.select(0, 0).select(0, y as i64).select(0, x as i64).fill_(pixel[0] as f32 / 255.0);
            tensor.select(0, 1).select(0, y as i64).select(0, x as i64).fill_(pixel[1] as f32 / 255.0);
            tensor.select(0, 2).select(0, y as i64).select(0, x as i64).fill_(pixel[2] as f32 / 255.0);
        }
    }
    
    Ok(tensor)
}

fn crop_face(img: &DynamicImage, detection: &Detection, padding_factor: f32, target_size: u32) -> Result<DynamicImage> {
    let (img_width, img_height) = img.dimensions();
    
    let x_min = detection.bbox[0] * img_width as f32;
    let y_min = detection.bbox[1] * img_height as f32;
    let x_max = detection.bbox[2] * img_width as f32;
    let y_max = detection.bbox[3] * img_height as f32;
    
    let center_x = (x_min + x_max) / 2.0;
    let center_y = (y_min + y_max) / 2.0;
    let width = x_max - x_min;
    let height = y_max - y_min;
    
    let size = width.max(height) * padding_factor;
    let half_size = size / 2.0;
    
    let crop_x_min = (center_x - half_size).max(0.0).min(img_width as f32) as u32;
    let crop_y_min = (center_y - half_size).max(0.0).min(img_height as f32) as u32;
    let crop_x_max = (center_x + half_size).max(0.0).min(img_width as f32) as u32;
    let crop_y_max = (center_y + half_size).max(0.0).min(img_height as f32) as u32;
    
    if crop_x_max <= crop_x_min || crop_y_max <= crop_y_min {
        anyhow::bail!("Invalid crop dimensions");
    }
    
    let cropped = img.crop_imm(
        crop_x_min,
        crop_y_min,
        crop_x_max - crop_x_min,
        crop_y_max - crop_y_min,
    );
    
    let resized = cropped.resize_exact(
        target_size,
        target_size,
        image::imageops::FilterType::Lanczos3,
    );
    
    Ok(resized)
}

fn process_batch(
    detector: &FaceDetector,
    image_paths: &[PathBuf],
    config: &Config,
    output_dir: &Path,
    face_counter: &mut usize,
) -> Result<usize> {
    let mut images = Vec::new();
    let mut valid_indices = Vec::new();
    
    for (i, path) in image_paths.iter().enumerate() {
        match image::open(path) {
            Ok(img) => {
                images.push(img);
                valid_indices.push(i);
            }
            Err(e) => {
                eprintln!("Error loading image {:?}: {}", path, e);
            }
        }
    }
    
    if images.is_empty() {
        return Ok(0);
    }
    
    let input_tensor = preprocess_images(&images, 640)?;
    
    let batch_detections = detector.detect(&input_tensor)?;
    
    let mut processed_faces = 0;
    
    for (idx, detections) in batch_detections.iter().enumerate() {
        let img_idx = valid_indices[idx];
        let image_path = &image_paths[img_idx];
        
        let mut sorted_detections = detections.clone();
        sorted_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let filtered_detections: Vec<_> = sorted_detections
            .into_iter()
            .filter(|d| d.confidence >= config.min_confidence)
            .collect();
        
        for (i, detection) in filtered_detections.iter().enumerate() {
            if *face_counter >= config.max_faces {
                return Ok(processed_faces);
            }
            
            match crop_face(&images[idx], detection, config.padding_factor, config.target_size) {
                Ok(cropped) => {
                    let stem = image_path.file_stem().unwrap_or_default().to_string_lossy();
                    let output_path = output_dir.join(format!(
                        "{}_face{}_conf{:.2}.png",
                        stem,
                        i,
                        detection.confidence * 100.0
                    ));
                    
                    cropped.save(&output_path)?;
                    
                    *face_counter += 1;
                    processed_faces += 1;
                }
                Err(e) => {
                    eprintln!("Error cropping face: {}", e);
                }
            }
        }
    }
    
    Ok(processed_faces)
}

fn main() -> Result<()> {
    let matches = App::new("Face Detection and Cropping")
        .version("1.0")
        .author("AI Assistant")
        .about("Detects and crops faces from images using YOLOv11")
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .value_name("FILE")
                .help("Sets a custom config file")
                .takes_value(true)
                .default_value("config.json"),
        )
        .get_matches();

    let config_path = matches.value_of("config").unwrap();
    let config_str = fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path))?;
    let config: Config = serde_json::from_str(&config_str)
        .with_context(|| format!("Failed to parse config file: {}", config_path))?;

    let output_dir = Path::new(&config.output_dir);
    fs::create_dir_all(output_dir)?;

    let detector = Arc::new(FaceDetector::new(&config.model_path)?);

    let image_paths: Vec<PathBuf> = fs::read_dir(&config.input_dir)?
        .filter_map(Result::ok)
        .filter(|entry| {
            let path = entry.path();
            path.is_file() && matches!(path.extension().and_then(|s| s.to_str()), 
                Some("jpg" | "jpeg" | "png" | "bmp"))
        })
        .map(|entry| entry.path())
        .collect();

    println!("Found {} images in input directory", image_paths.len());

    let progress_bar = ProgressBar::new(config.max_faces as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} faces | {eta}")?
            .progress_chars("##-"),
    );

    let mut face_counter = 0;
    let batches: Vec<_> = image_paths
        .chunks(config.batch_size)
        .collect();

    for batch in batches {
        if face_counter >= config.max_faces {
            break;
        }
        
        let processed = process_batch(
            &detector,
            batch,
            &config,
            output_dir,
            &mut face_counter,
        )?;
        
        progress_bar.inc(processed as u64);
    }

    progress_bar.finish_with_message(format!("Processed {} faces", face_counter));
    println!("Face extraction complete. {} faces extracted to {}", 
             face_counter, config.output_dir);

    Ok(())
}