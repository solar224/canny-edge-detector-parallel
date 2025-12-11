# Canny Edge Detector with Pthread Parallelization

A Canny edge detector written in C++ which only uses OpenCV for file reading/writing. The canny edge detection implements the following without any external libraries: Gaussian blur, Sobel filter as the edge detection operator, non-maximum suppression, and double thresholding.

**This version includes pthread parallelization for improved performance (1-6 threads).**

## Example

### Original Image
<img src="images/Sukuna.jpg" alt="Original Image" width="400">

### Image after Canny Edge Detection
<img src="images/SukunaCanny.jpg" alt="Image after Canny Edge Detection" width="400">

---

## Prerequisites

Before building, ensure you have the following installed:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install cmake libopencv-dev build-essential
```

---

## Build Instructions

### Step 1: Clone the repository
```bash
git clone <repository-url>
cd canny-edge-detector
```

### Step 2: Create build directory
```bash
mkdir build
cd build
```

### Step 3: Configure with CMake
```bash
cmake ..
```

### Step 4: Build the project
```bash
make -j4
```

This will generate two executables:
- `canny` - Main edge detection program
- `benchmark` - Performance benchmarking tool

---

## Usage

### Basic Usage (Single Thread)
```bash
./canny
```
This processes `../images/Sukuna.jpg` and saves the result to `../images/SukunaCanny.jpg`.

### Parallel Execution (Specify Thread Count)
```bash
./canny <num_threads>
```

**Examples:**
```bash
./canny 1    # Single thread
./canny 4    # 4 threads
./canny 6    # 6 threads (maximum recommended)
```

### Custom Input/Output
```bash
./canny <num_threads> <input_image> <output_image>
```

**Examples:**
```bash
./canny 6 ../images/Sukuna.jpg ../images/output.jpg
./canny 4 /path/to/input.jpg /path/to/output.jpg
```

---

## Benchmark

Run the benchmark tool to measure performance across different thread counts:

```bash
./benchmark
```

This will output three tables:

### Table 1: Overall Performance
Shows total execution time and speedup for 1-6 threads.

### Table 2: Per-Function Performance
Shows timing breakdown for each function:
- Gaussian blur
- RGB to Grayscale conversion
- Canny filter

### Table 3: Different Noise Levels
Tests performance with different noise levels:
- `sukuna_noise0.jpg` (original)
- `Sukuna_noise1_gauss15.jpg` (Gaussian noise σ=15)
- `Sukuna_noise2_gauss30.jpg` (Gaussian noise σ=30)

**Sample Output:**
```
================================================================================
Table 1: Overall Performance
================================================================================
   Threads        Average time (ms)      Speedup (= T1 / Tn)
--------------------------------------------------------------------------------
         1                  2030.00                     1.00
         2                   965.00                     2.10
         3                   670.00                     3.03
         4                   540.00                     3.76
         5                   460.00                     4.41
         6                   410.00                     4.95
================================================================================
```

---

## Project Structure

```
canny-edge-detector/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── canny.h                 # Original header (serial version)
├── canny.cpp               # Original implementation (serial version)
├── canny_parallel.h        # Parallel version header
├── canny_parallel.cpp      # Pthread parallel implementation
├── main.cpp                # Main program entry point
├── benchmark.cpp           # Benchmarking tool
├── images/
│   ├── Sukuna.jpg          # Sample input image
│   └── SukunaCanny.jpg     # Sample output image
└── build/                  # Build directory (created after cmake)
    ├── canny               # Main executable
    └── benchmark           # Benchmark executable
```

---

## Algorithm Overview

The Canny edge detection algorithm consists of the following steps:

1. **Gaussian Blur** - Reduces noise using a 5x5 Gaussian kernel
2. **Grayscale Conversion** - Converts RGB to grayscale
3. **Sobel Filter** - Computes gradient magnitude and direction
4. **Non-Maximum Suppression** - Thins edges to 1-pixel width
5. **Double Thresholding** - Classifies edges as strong, weak, or non-edges
6. **Hysteresis** - Connects weak edges to strong edges

### Parallelization Strategy

The implementation uses **row-based parallelization** with pthread:

| Function | Strategy | Scalability |
|----------|----------|-------------|
| Gaussian Blur | Row partitioning | Excellent (near-linear) |
| RGB to Grayscale | Row partitioning | Good |
| Canny Filter | Phase-based parallel | Moderate (has sequential phase) |

---

## Parameters

You can adjust edge detection sensitivity in `main.cpp`:

```cpp
double lowerThreshold = 0.03;   // Lower threshold ratio
double higherThreshold = 0.1;   // Higher threshold ratio
```

- **Lower values** → More edges detected
- **Higher values** → Fewer edges detected

---

## License

This project is open source. Feel free to use and modify as needed.
