# Canny Edge Detector with Pthread Parallelization

This project implements a high-performance Canny Edge Detector in C++ from scratch, utilizing Pthreads for row-based parallel processing. It demonstrates manual implementation of core computer vision concepts—including Gaussian blur, gradient computation (Sobel), non-maximum suppression, and double thresholding—without relying on external libraries for the algorithms themselves. OpenCV is used strictly for reading and writing image files.

---

## Installation

### Using Pre-built Binaries (Recommended)

Download the latest release from the [Releases page](https://github.com/solar224/canny-edge-detector-parallel/releases):

1. Download the `canny-edge-detector-vX.X.X-linux-x64.tar.gz` file
2. Extract the archive:
   ```bash
   tar -xzf canny-edge-detector-vX.X.X-linux-x64.tar.gz
   ```
3. Make sure you have OpenCV installed:
   ```bash
   # Ubuntu/Debian
   sudo apt install libopencv-dev
   ```
4. Run the executables:
   ```bash
   ./canny
   ./benchmark
   ```

### Building from Source

If you prefer to build from source, follow the build instructions below.

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

## Creating Releases

This project uses automated GitHub Actions workflows to create releases. Here's how to create a new release:

### For Maintainers

1. **Update version in CMakeLists.txt** (if needed):
   ```cmake
   project(canny VERSION X.Y.Z)
   ```

2. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Prepare for release vX.Y.Z"
   git push
   ```

3. **Create and push a version tag**:
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```

4. **Automated workflow will**:
   - Build the project with all dependencies
   - Create compiled binaries for Linux x64
   - Package everything into a tar.gz archive
   - Create a GitHub Release with the binaries
   - Generate release notes automatically

### Versioning

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): New functionality in a backwards-compatible manner
- **PATCH** version (0.0.X): Backwards-compatible bug fixes

---

## License

This project is open source. Feel free to use and modify as needed.
