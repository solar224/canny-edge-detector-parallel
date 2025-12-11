#pragma once

#include <math.h>
#include <pthread.h>

#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>

// Thread data structure for passing parameters
struct ThreadData {
    int threadId;
    int numThreads;
    int startRow;
    int endRow;
    int sizeRows;
    int sizeCols;
    int sizeDepth;
    std::vector<int>* inputPixels;
    std::vector<int>* outputPixels;
    std::vector<std::vector<double>>* kernel;
    double kernelConst;
    // For cannyFilter
    double* G;
    std::vector<int>* theta;
    double lowerThreshold;
    double higherThreshold;
    double* largestG;
    pthread_mutex_t* mutex;
    pthread_barrier_t* barrier;
};

// Global thread count setter
void setNumThreads(int n);
int getNumThreads();

// Parallel versions of the main functions
std::vector<int> gaussianBlur_parallel(std::vector<int>& pixels, std::vector<std::vector<double>>& kernel, 
                                        double kernelConst, int sizeRows, int sizeCols, int sizeDepth);
std::vector<int> rgbToGrayscale_parallel(std::vector<int>& pixels, int sizeRows, int sizeCols, int sizeDepth);
std::vector<int> cannyFilter_parallel(std::vector<int>& pixels, int sizeRows, int sizeCols, int sizeDepth, 
                                       double lowerThreshold, double higherThreshold);

// Parallel version of the main canny edge detection function
void cannyEdgeDetection_parallel(std::string readLocation, std::string writeLocation, 
                                  double lowerThreshold, double higherThreshold);

// Benchmark utilities
double getCurrentTimeMs();
