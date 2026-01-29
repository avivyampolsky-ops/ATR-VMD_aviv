# ATR-VMD

**ATR-VMD** (Automatic Target Recognition - Video Motion Detection) is a high-performance system designed for detecting and tracking moving objects in video streams. It leverages frame registration (translation or homography), background subtraction (MOG2), and Kalman filter tracking to provide robust object detection and tracking.

The system is optimized for performance, featuring C++ implementations for critical components and CUDA acceleration for image processing and motion estimation.

## Features

*   **Frame Registration**: Aligns frames to a common reference to compensate for camera motion (using Homography or Phase Correlation).
*   **Motion Detection**: Uses MOG2 (Mixture of Gaussians) background subtraction to detect moving objects.
*   **Tracking**: Implements a Kalman Filter with IoU (Intersection over Union) association for tracking objects across frames.
*   **Performance**: Core modules are implemented in C++ with CUDA support for GPU acceleration.
*   **Debug Tools**: Comprehensive visualization and logging tools for analyzing tracking performance.

## Prerequisites

### System Requirements

*   **OpenCV 4.x** with **CUDA** support is required.
*   The following OpenCV modules must be available:
    *   `opencv_core`, `opencv_imgproc`, `opencv_video`, `opencv_calib3d`
    *   `opencv_features2d`, `opencv_flann`
    *   `opencv_cudaarithm`, `opencv_cudabgsegm`, `opencv_cudafilters`, `opencv_cudaimgproc`, `opencv_cudawarping`
    *   `opencv_xfeatures2d`

### Python Requirements

*   Python 3.x
*   `numpy`
*   `PyYAML`
*   `tqdm`
*   `setuptools`
*   `pybind11`
*   `opencv-python` (Note: Ensure this does not conflict with your system's OpenCV+CUDA installation if you are relying on it, though the python code uses `cv2` which typically comes from this package or the system).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Python dependencies**:
    ```bash
    pip install numpy PyYAML tqdm setuptools pybind11
    ```

3.  **Build C++ Extensions**:
    The project uses `pybind11` to interface with the C++/CUDA modules. Build the extensions in-place using the provided setup script:
    ```bash
    python modules/setup_translation_gpu_cpp.py build_ext --inplace
    ```

## Configuration

The system is configured via the `config.yaml` file. This file controls various aspects of the pipeline, including:

*   **General**: Toggle C++ / CUDA usage.
*   **Debug**: Enable/disable debug output, visualization, and timing logs.
*   **Registration**: Choose mode (`homography` or `translation`), set parameters for feature matching (FAST/BRIEF, KNN) or phase correlation.
*   **Detection**: Adjust learning rates and thresholds for MOG2.
*   **Tracker**: Configure Kalman filter parameters (IoU thresholds, max lost frames, etc.).

## Usage

The main entry point is `main.py`. You can run it with default settings or specify input data and configuration paths.

### Basic Command

```bash
python main.py --data-path /path/to/video.mp4 --out-dir /path/to/output
```

### Arguments

*   `--data-path`: Path to the input video file or directory of images.
*   `--out-dir`: Directory where results (debug videos, logs) will be saved.
*   `--config-path`: Path to the YAML configuration file (defaults to `config.yaml` in the root).

### Example

```bash
python main.py \
    --data-path ./data/test_video.mkv \
    --out-dir ./results \
    --config-path ./config.yaml
```

## Project Structure

*   `main.py`: Entry point of the application. Handles initialization and the main processing loop.
*   `config.yaml`: Default configuration file.
*   `modules/`: Contains Python and C++ source code.
    *   `atr_vmd.py`: Main tracking class coordinating registration, detection, and tracking.
    *   `setup_translation_gpu_cpp.py`: Setup script for building C++ extensions.
    *   `*_cpp.cpp`: C++ implementations for performance-critical tasks.
    *   `debug_utils.py`: Utilities for visualization and logging.
    *   `config_loader.py`: Helper to parse YAML configuration.
