#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>

#include "canny.h"
#include "canny_parallel.h"

// Benchmark configuration
const int NUM_RUNS = 10;  // Number of runs to average
const int MAX_THREADS = 6;

struct BenchmarkResult {
    double gaussianTime;
    double grayscaleTime;
    double cannyTime;
    double totalTime;
};

// Generate noisy image using OpenCV
void generateNoisyImages(const std::string& basePath) {
    cv::Mat img = cv::imread(basePath);
    if (img.empty()) {
        std::cerr << "Error: Could not read base image\n";
        return;
    }
    
    // noise0 - original image (no noise)
    cv::imwrite("../images/sukuna_noise0.jpg", img);
    
    // noise1 - add Gaussian noise with sigma=15
    cv::Mat noise1 = img.clone();
    cv::Mat gaussianNoise1(img.size(), img.type());
    cv::randn(gaussianNoise1, 0, 15);
    cv::add(noise1, gaussianNoise1, noise1);
    cv::imwrite("../images/Sukuna_noise1_gauss15.jpg", noise1);
    
    // noise2 - add Gaussian noise with sigma=30
    cv::Mat noise2 = img.clone();
    cv::Mat gaussianNoise2(img.size(), img.type());
    cv::randn(gaussianNoise2, 0, 30);
    cv::add(noise2, gaussianNoise2, noise2);
    cv::imwrite("../images/Sukuna_noise2_gauss30.jpg", noise2);
    
    std::cout << "Generated test images:\n";
    std::cout << "  - sukuna_noise0.jpg (original)\n";
    std::cout << "  - Sukuna_noise1_gauss15.jpg (noise sigma=15)\n";
    std::cout << "  - Sukuna_noise2_gauss30.jpg (noise sigma=30)\n\n";
}

// Benchmark a single run with specified threads
BenchmarkResult runSingleBenchmark(const std::string& imagePath, int numThreads) {
    setNumThreads(numThreads);
    BenchmarkResult result = {0, 0, 0, 0};
    
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        return result;
    }
    
    uint8_t* pixelPtr = (uint8_t*)img.data;
    int sizeRows = img.rows;
    int sizeCols = img.cols;
    int sizeDepth = img.channels();
    
    std::vector<std::vector<double>> kernel = {{2.0, 4.0, 5.0, 4.0, 2.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {5.0, 12.0, 15.0, 12.0, 5.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);
    double lowerThreshold = 0.03;
    double higherThreshold = 0.1;
    
    std::vector<int> pixels = imgToArray(img, pixelPtr, sizeRows, sizeCols, sizeDepth);
    
    // Benchmark Gaussian Blur
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> pixelsBlur = gaussianBlur_parallel(pixels, kernel, kernelConst, sizeRows, sizeCols, sizeDepth);
    auto end = std::chrono::high_resolution_clock::now();
    result.gaussianTime = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Benchmark RGB to Grayscale
    start = std::chrono::high_resolution_clock::now();
    std::vector<int> pixelsGray = rgbToGrayscale_parallel(pixelsBlur, sizeRows, sizeCols, sizeDepth);
    end = std::chrono::high_resolution_clock::now();
    result.grayscaleTime = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Benchmark Canny Filter
    start = std::chrono::high_resolution_clock::now();
    std::vector<int> pixelsCanny = cannyFilter_parallel(pixelsGray, sizeRows, sizeCols, 1, lowerThreshold, higherThreshold);
    end = std::chrono::high_resolution_clock::now();
    result.cannyTime = std::chrono::duration<double, std::milli>(end - start).count();
    
    result.totalTime = result.gaussianTime + result.grayscaleTime + result.cannyTime;
    
    return result;
}

// Run multiple benchmarks and average
BenchmarkResult runAverageBenchmark(const std::string& imagePath, int numThreads) {
    BenchmarkResult avg = {0, 0, 0, 0};
    
    for (int run = 0; run < NUM_RUNS; run++) {
        BenchmarkResult r = runSingleBenchmark(imagePath, numThreads);
        avg.gaussianTime += r.gaussianTime;
        avg.grayscaleTime += r.grayscaleTime;
        avg.cannyTime += r.cannyTime;
        avg.totalTime += r.totalTime;
    }
    
    avg.gaussianTime /= NUM_RUNS;
    avg.grayscaleTime /= NUM_RUNS;
    avg.cannyTime /= NUM_RUNS;
    avg.totalTime /= NUM_RUNS;
    
    return avg;
}

void printTable1(const std::string& imagePath) {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "Table 1: Overall Performance\n";
    std::cout << "================================================================================\n";
    std::cout << std::setw(10) << "Threads" 
              << std::setw(25) << "Average time (ms)" 
              << std::setw(25) << "Speedup (= T1 / Tn)" << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    
    double baseTime = runAverageBenchmark(imagePath, 1).totalTime;
    
    for (int t = 1; t <= MAX_THREADS; t++) {
        BenchmarkResult r = runAverageBenchmark(imagePath, t);
        double speedup = baseTime / r.totalTime;
        std::cout << std::setw(10) << t 
                  << std::setw(25) << std::fixed << std::setprecision(2) << r.totalTime
                  << std::setw(25) << std::fixed << std::setprecision(2) << speedup << "\n";
    }
    std::cout << "================================================================================\n";
}

void printTable2(const std::string& imagePath) {
    std::cout << "\n";
    std::cout << "========================================================================================================\n";
    std::cout << "Table 2: Per-Function Performance\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::setw(10) << "Threads" 
              << std::setw(22) << "Gaussian blur only"
              << std::setw(22) << "rgbToGrayscale only"
              << std::setw(22) << "cannyFilter only"
              << std::setw(22) << "Speedup (= T1 / Tn)" << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";
    
    BenchmarkResult base = runAverageBenchmark(imagePath, 1);
    
    for (int t = 1; t <= MAX_THREADS; t++) {
        BenchmarkResult r = runAverageBenchmark(imagePath, t);
        double speedup = base.totalTime / r.totalTime;
        std::cout << std::setw(10) << t 
                  << std::setw(22) << std::fixed << std::setprecision(2) << r.gaussianTime
                  << std::setw(22) << std::fixed << std::setprecision(2) << r.grayscaleTime
                  << std::setw(22) << std::fixed << std::setprecision(2) << r.cannyTime
                  << std::setw(22) << std::fixed << std::setprecision(2) << speedup << "\n";
    }
    std::cout << "========================================================================================================\n";
}

void printTable3() {
    std::string noise0 = "../images/sukuna_noise0.jpg";
    std::string noise1 = "../images/Sukuna_noise1_gauss15.jpg";
    std::string noise2 = "../images/Sukuna_noise2_gauss30.jpg";
    
    std::cout << "\n";
    std::cout << "========================================================================================================\n";
    std::cout << "Table 3: Different Noise Levels\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::setw(10) << "Threads" 
              << std::setw(22) << "sukuna_noise0(ms)"
              << std::setw(28) << "Sukuna_noise1_gauss15 (ms)"
              << std::setw(28) << "Sukuna_noise2_gauss30 (ms)"
              << std::setw(22) << "Speedup (= T1 / Tn)" << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";
    
    double baseTime0 = runAverageBenchmark(noise0, 1).totalTime;
    
    for (int t = 1; t <= MAX_THREADS; t++) {
        BenchmarkResult r0 = runAverageBenchmark(noise0, t);
        BenchmarkResult r1 = runAverageBenchmark(noise1, t);
        BenchmarkResult r2 = runAverageBenchmark(noise2, t);
        double speedup = baseTime0 / r0.totalTime;
        
        std::cout << std::setw(10) << t 
                  << std::setw(22) << std::fixed << std::setprecision(2) << r0.totalTime
                  << std::setw(28) << std::fixed << std::setprecision(2) << r1.totalTime
                  << std::setw(28) << std::fixed << std::setprecision(2) << r2.totalTime
                  << std::setw(22) << std::fixed << std::setprecision(2) << speedup << "\n";
    }
    std::cout << "========================================================================================================\n";
}

int main(int argc, char* argv[]) {
    std::string imagePath = "../images/Sukuna.jpg";
    
    if (argc > 1) {
        imagePath = argv[1];
    }
    
    std::cout << "================================================================================\n";
    std::cout << "        CANNY EDGE DETECTOR - PTHREAD BENCHMARK (10 runs average)\n";
    std::cout << "================================================================================\n";
    std::cout << "Image: " << imagePath << "\n";
    std::cout << "Runs per test: " << NUM_RUNS << "\n";
    
    // Generate noisy test images
    std::cout << "\nGenerating noisy test images...\n";
    generateNoisyImages(imagePath);
    
    // Table 1: Overall performance
    std::cout << "\nRunning Table 1 benchmarks...\n";
    printTable1(imagePath);
    
    // Table 2: Per-function performance
    std::cout << "\nRunning Table 2 benchmarks...\n";
    printTable2(imagePath);
    
    // Table 3: Different noise levels
    std::cout << "\nRunning Table 3 benchmarks...\n";
    printTable3();
    
    std::cout << "\nBenchmark complete!\n";
    
    return 0;
}
