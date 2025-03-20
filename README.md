# Face Detection and Cropping Pipeline

A Rust-based tool for creating a dataset of aligned face images using YOLOv11-face detection model.

Note: I have used only two images for face detection, but I built this project with rayon, which enables parallel processing. It can process an entire folder from the dataset. I used only a small sample because processing 4,000 to 10,000 images is quite computationally intensive.

## Overview

This program processes batches of images to detect, crop, and save face regions. It's designed for creating large datasets of face images (4,000-10,000) while handling various edge cases efficiently. The pipeline leverages GPU acceleration when available and includes parallel processing for optimal performance.

## Features

- High-accuracy face detection using YOLOv11-face
- Batch processing for efficient GPU utilization
- Parallel execution for faster processing
- Configurable face cropping with padding
- Confidence-based filtering of detections
- Non-maximum suppression to avoid duplicate faces
- Progress tracking with ETA
- Robust error handling for corrupted images

## Prerequisites

- Rust (1.65 or newer)
- LibTorch (PyTorch C++ API)
- CUDA toolkit (optional, for GPU acceleration)
- YOLOv11-face model in TorchScript format

## Installation

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Install system dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential pkg-config cmake

# Fedora
sudo dnf install gcc gcc-c++ make cmake pkg-config

# macOS
brew install cmake pkg-config
```

### 3. Install LibTorch

```bash
# Create directory for libraries
mkdir -p libs
cd libs

# For CUDA support (adjust for your CUDA version)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118.zip

# For CPU only
# wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
# unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

# Set environment variables
export LIBTORCH=$PWD/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cd ..
```

### 4. Set up YOLOv11-face model

```bash
# Create Python virtual environment
python -m venv face_detection_env
source face_detection_env/bin/activate  # On Linux/Mac
# face_detection_env\Scripts\activate   # On Windows

# Install YOLOv11-face dependencies
pip install ultralytics

# Download YOLOv11 
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# Download pre-trained weights if needed
# (Check repository for download links or use provided weights)

# Convert model to TorchScript format
mkdir -p ../models
python -c "from ultralytics import YOLO; model = YOLO('path/to/yolov11n.pt'); model.export(format='torchscript', imgsz=640)"
mv yolov11n.torchscript ../models/

cd ..
deactivate  # Exit virtual environment when done
```

## Project Setup

### 1. Create Rust project

```bash
cargo new face_detection_pipeline
cd face_detection_pipeline
```

### 2. Configure dependencies in Cargo.toml

Replace the contents of Cargo.toml with:

```toml
[package]
name = "face_detection_pipeline"
version = "0.1.0"
edition = "2021"
authors = ["Your Name"]
description = "A face detection and cropping pipeline using YOLOv11"

[dependencies]
tch = "0.10.3"        # PyTorch bindings for Rust
image = "0.24.6"      # Image processing
anyhow = "1.0.71"     # Error handling
rayon = "1.7.0"       # Parallel processing
clap = "3.2.25"       # Command line argument parsing
serde = { version = "1.0.163", features = ["derive"] }
serde_json = "1.0.96" # JSON serialization/deserialization
indicatif = "0.17.3"  # Progress bars
thiserror = "1.0.40"  # Error definitions
```

### 3. Create config.json file

Create a file named `config.json` in the project root:

```json
{
  "input_dir": "./dataset/wider_face/images",
  "output_dir": "./output/faces",
  "model_path": "./models/yolov11n.torchscript",
  "min_confidence": 0.5,
  "target_size": 224,
  "padding_factor": 1.5,
  "max_faces": 10000,
  "batch_size": 16
}
```

Adjust the paths and settings according to your setup.

### 4. Prepare input dataset

```bash
# Option 1: Download WIDER FACE dataset
mkdir -p dataset/wider_face
cd dataset/wider_face
wget http://shuoyang1213.me/WIDERFACE/WIDER_train.zip
unzip WIDER_train.zip
cd ../..

# Option 2: Use your own images
mkdir -p dataset/my_images
# Copy your images to this directory
```

### 5. Create output directory

```bash
mkdir -p output/faces
```

### 6. Configure build for CUDA (if needed)

Create or edit `.cargo/config.toml`:

```toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "link-arg=-Wl,-rpath,/path/to/libtorch/lib"]
```

Replace `/path/to/libtorch/lib` with your actual LibTorch lib path.

## Building and Running

```bash
# Make sure LibTorch is in your path
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Build in release mode
cargo build --release

# Run with config file
./target/release/face_detection_pipeline -c config.json
```

## Configuration Options

- `input_dir`: Directory containing input images
- `output_dir`: Directory where cropped faces will be saved
- `model_path`: Path to YOLOv11-face TorchScript model
- `min_confidence`: Minimum detection confidence (0.0-1.0)
- `target_size`: Output image size in pixels (square)
- `padding_factor`: Padding around face (1.0 = no padding, 1.5 = 50% padding)
- `max_faces`: Maximum number of faces to extract
- `batch_size`: Number of images to process in each batch

## Troubleshooting

### CUDA Not Found

If you have issues with CUDA detection:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### LibTorch Linking Issues

Ensure you have the correct version of LibTorch for your CUDA version. You can check CUDA version with:

```bash
nvcc --version
```

### Build Failures

Make sure all dependencies are installed and the LibTorch path is correctly set:

```bash
cargo clean
export LIBTORCH=/absolute/path/to/libtorch
cargo build --release
```

### Memory Issues

If you run out of GPU memory, reduce the batch size in the config file.

## Performance Optimization

- Increase `batch_size` for better GPU utilization (if memory allows)
- Set `min_confidence` higher to reduce processing time spent on low-quality detections
- For CPUs with many cores, the program will automatically use parallel processing

## Output Format

Faces are saved as PNG files with filenames in the format:
```
{original_image_name}_face{index}_conf{confidence}.png
```

Example: `image1_face0_conf95.23.png`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
