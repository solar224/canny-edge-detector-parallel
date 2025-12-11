#include "canny_parallel.h"
#include "canny.h"
#include <chrono>
#include <algorithm>
#include <cstring>

static int g_numThreads = 1;

void setNumThreads(int n) {
    g_numThreads = std::max(1, std::min(n, 16));
}

int getNumThreads() {
    return g_numThreads;
}

double getCurrentTimeMs() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

// ============================================================================
// GAUSSIAN BLUR - PARALLEL VERSION
// ============================================================================

void* gaussianBlurWorker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    for (int i = data->startRow; i < data->endRow; i++) {
        for (int j = 0; j < data->sizeCols; j++) {
            for (int k = 0; k < data->sizeDepth; k++) {
                double sum = 0;
                double sumKernel = 0;
                for (int y = -2; y <= 2; y++) {
                    for (int x = -2; x <= 2; x++) {
                        if ((i + x) >= 0 && (i + x) < data->sizeRows && 
                            (j + y) >= 0 && (j + y) < data->sizeCols) {
                            double channel = (double)(*data->inputPixels)[(i + x) * data->sizeCols * data->sizeDepth + 
                                                                           (j + y) * data->sizeDepth + k];
                            sum += channel * data->kernelConst * (*data->kernel)[x + 2][y + 2];
                            sumKernel += data->kernelConst * (*data->kernel)[x + 2][y + 2];
                        }
                    }
                }
                (*data->outputPixels)[i * data->sizeCols * data->sizeDepth + j * data->sizeDepth + k] = 
                    (int)(sum / sumKernel);
            }
        }
    }
    return nullptr;
}

std::vector<int> gaussianBlur_parallel(std::vector<int>& pixels, std::vector<std::vector<double>>& kernel, 
                                        double kernelConst, int sizeRows, int sizeCols, int sizeDepth) {
    std::vector<int> pixelsBlur(sizeRows * sizeCols * sizeDepth);
    int numThreads = g_numThreads;
    
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    
    int rowsPerThread = sizeRows / numThreads;
    
    for (int t = 0; t < numThreads; t++) {
        threadData[t].threadId = t;
        threadData[t].numThreads = numThreads;
        threadData[t].startRow = t * rowsPerThread;
        threadData[t].endRow = (t == numThreads - 1) ? sizeRows : (t + 1) * rowsPerThread;
        threadData[t].sizeRows = sizeRows;
        threadData[t].sizeCols = sizeCols;
        threadData[t].sizeDepth = sizeDepth;
        threadData[t].inputPixels = &pixels;
        threadData[t].outputPixels = &pixelsBlur;
        threadData[t].kernel = &kernel;
        threadData[t].kernelConst = kernelConst;
        
        pthread_create(&threads[t], nullptr, gaussianBlurWorker, &threadData[t]);
    }
    
    for (int t = 0; t < numThreads; t++) {
        pthread_join(threads[t], nullptr);
    }
    
    return pixelsBlur;
}

// ============================================================================
// RGB TO GRAYSCALE - PARALLEL VERSION
// ============================================================================

void* rgbToGrayscaleWorker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    for (int i = data->startRow; i < data->endRow; i++) {
        for (int j = 0; j < data->sizeCols; j++) {
            int sum = 0;
            for (int k = 0; k < data->sizeDepth; k++) {
                sum += (*data->inputPixels)[i * data->sizeCols * data->sizeDepth + j * data->sizeDepth + k];
            }
            (*data->outputPixels)[i * data->sizeCols + j] = (int)(sum / data->sizeDepth);
        }
    }
    return nullptr;
}

std::vector<int> rgbToGrayscale_parallel(std::vector<int>& pixels, int sizeRows, int sizeCols, int sizeDepth) {
    std::vector<int> pixelsGray(sizeRows * sizeCols);
    int numThreads = g_numThreads;
    
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    
    int rowsPerThread = sizeRows / numThreads;
    
    for (int t = 0; t < numThreads; t++) {
        threadData[t].threadId = t;
        threadData[t].numThreads = numThreads;
        threadData[t].startRow = t * rowsPerThread;
        threadData[t].endRow = (t == numThreads - 1) ? sizeRows : (t + 1) * rowsPerThread;
        threadData[t].sizeRows = sizeRows;
        threadData[t].sizeCols = sizeCols;
        threadData[t].sizeDepth = sizeDepth;
        threadData[t].inputPixels = &pixels;
        threadData[t].outputPixels = &pixelsGray;
        
        pthread_create(&threads[t], nullptr, rgbToGrayscaleWorker, &threadData[t]);
    }
    
    for (int t = 0; t < numThreads; t++) {
        pthread_join(threads[t], nullptr);
    }
    
    return pixelsGray;
}

// ============================================================================
// CANNY FILTER - PARALLEL VERSION (Most Complex)
// ============================================================================

// Phase 1: Compute gradient magnitude and direction
void* cannyPhase1Worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    double localLargestG = 0;
    
    // Compute start/end with boundary handling (skip edges)
    int startRow = std::max(1, data->startRow);
    int endRow = std::min(data->sizeRows - 1, data->endRow);
    
    for (int i = startRow; i < endRow; i++) {
        for (int j = 1; j < data->sizeCols - 1; j++) {
            double gxValue = 0;
            double gyValue = 0;
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    gxValue += gx[1 - x][1 - y] * (double)((*data->inputPixels)[(i + x) * data->sizeCols + j + y]);
                    gyValue += gy[1 - x][1 - y] * (double)((*data->inputPixels)[(i + x) * data->sizeCols + j + y]);
                }
            }
            
            data->G[i * data->sizeCols + j] = std::sqrt(gxValue * gxValue + gyValue * gyValue);
            double atanResult = atan2(gyValue, gxValue) * 180.0 / 3.14159265;
            (*data->theta)[i * data->sizeCols + j] = (int)(180.0 + atanResult);
            (*data->theta)[i * data->sizeCols + j] = ((*data->theta)[i * data->sizeCols + j] / 45) * 45;
            
            if (data->G[i * data->sizeCols + j] > localLargestG) {
                localLargestG = data->G[i * data->sizeCols + j];
            }
        }
    }
    
    // Update global largestG with mutex
    pthread_mutex_lock(data->mutex);
    if (localLargestG > *(data->largestG)) {
        *(data->largestG) = localLargestG;
    }
    pthread_mutex_unlock(data->mutex);
    
    return nullptr;
}

// Phase 2: Non-maximum suppression
void* cannyPhase2Worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    int startRow = std::max(1, data->startRow);
    int endRow = std::min(data->sizeRows - 1, data->endRow);
    double largestG = *(data->largestG);
    
    for (int i = startRow; i < endRow; i++) {
        for (int j = 1; j < data->sizeCols - 1; j++) {
            int theta = (*data->theta)[i * data->sizeCols + j];
            double currentG = data->G[i * data->sizeCols + j];
            
            if (theta == 0 || theta == 180) {
                if (currentG < data->G[i * data->sizeCols + j - 1] || 
                    currentG < data->G[i * data->sizeCols + j + 1]) {
                    data->G[i * data->sizeCols + j] = 0;
                }
            } else if (theta == 45 || theta == 225) {
                if (currentG < data->G[(i + 1) * data->sizeCols + j + 1] || 
                    currentG < data->G[(i - 1) * data->sizeCols + j - 1]) {
                    data->G[i * data->sizeCols + j] = 0;
                }
            } else if (theta == 90 || theta == 270) {
                if (currentG < data->G[(i + 1) * data->sizeCols + j] || 
                    currentG < data->G[(i - 1) * data->sizeCols + j]) {
                    data->G[i * data->sizeCols + j] = 0;
                }
            } else {
                if (currentG < data->G[(i + 1) * data->sizeCols + j - 1] || 
                    currentG < data->G[(i - 1) * data->sizeCols + j + 1]) {
                    data->G[i * data->sizeCols + j] = 0;
                }
            }
            
            (*data->outputPixels)[i * data->sizeCols + j] = 
                (int)(data->G[i * data->sizeCols + j] * (255.0 / largestG));
        }
    }
    
    return nullptr;
}

std::vector<int> cannyFilter_parallel(std::vector<int>& pixels, int sizeRows, int sizeCols, int sizeDepth, 
                                       double lowerThreshold, double higherThreshold) {
    std::vector<int> pixelsCanny(sizeRows * sizeCols, 0);
    double* G = new double[sizeRows * sizeCols]();
    std::vector<int> theta(sizeRows * sizeCols, 0);
    double largestG = 0;
    
    int numThreads = g_numThreads;
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    
    int rowsPerThread = sizeRows / numThreads;
    
    // Initialize thread data
    for (int t = 0; t < numThreads; t++) {
        threadData[t].threadId = t;
        threadData[t].numThreads = numThreads;
        threadData[t].startRow = t * rowsPerThread;
        threadData[t].endRow = (t == numThreads - 1) ? sizeRows : (t + 1) * rowsPerThread;
        threadData[t].sizeRows = sizeRows;
        threadData[t].sizeCols = sizeCols;
        threadData[t].sizeDepth = sizeDepth;
        threadData[t].inputPixels = &pixels;
        threadData[t].outputPixels = &pixelsCanny;
        threadData[t].G = G;
        threadData[t].theta = &theta;
        threadData[t].lowerThreshold = lowerThreshold;
        threadData[t].higherThreshold = higherThreshold;
        threadData[t].largestG = &largestG;
        threadData[t].mutex = &mutex;
    }
    
    // Phase 1: Compute gradients (parallel)
    for (int t = 0; t < numThreads; t++) {
        pthread_create(&threads[t], nullptr, cannyPhase1Worker, &threadData[t]);
    }
    for (int t = 0; t < numThreads; t++) {
        pthread_join(threads[t], nullptr);
    }
    
    // Handle edge pixels (copy from neighbors) - single thread
    for (int j = 1; j < sizeCols - 1; j++) {
        G[j] = G[sizeCols + j];
        theta[j] = theta[sizeCols + j];
        G[(sizeRows - 1) * sizeCols + j] = G[(sizeRows - 2) * sizeCols + j];
        theta[(sizeRows - 1) * sizeCols + j] = theta[(sizeRows - 2) * sizeCols + j];
    }
    for (int i = 0; i < sizeRows; i++) {
        G[i * sizeCols] = G[i * sizeCols + 1];
        theta[i * sizeCols] = theta[i * sizeCols + 1];
        G[i * sizeCols + sizeCols - 1] = G[i * sizeCols + sizeCols - 2];
        theta[i * sizeCols + sizeCols - 1] = theta[i * sizeCols + sizeCols - 2];
    }
    
    // Phase 2: Non-maximum suppression (parallel)
    for (int t = 0; t < numThreads; t++) {
        pthread_create(&threads[t], nullptr, cannyPhase2Worker, &threadData[t]);
    }
    for (int t = 0; t < numThreads; t++) {
        pthread_join(threads[t], nullptr);
    }
    
    // Phase 3: Double thresholding (sequential due to dependencies)
    bool changes;
    do {
        changes = false;
        for (int i = 1; i < sizeRows - 1; i++) {
            for (int j = 1; j < sizeCols - 1; j++) {
                if (G[i * sizeCols + j] < (lowerThreshold * largestG)) {
                    G[i * sizeCols + j] = 0;
                } else if (G[i * sizeCols + j] >= (higherThreshold * largestG)) {
                    continue;
                } else {
                    double tempG = G[i * sizeCols + j];
                    G[i * sizeCols + j] = 0;
                    for (int x = -1; x <= 1; x++) {
                        bool breakLoop = false;
                        for (int y = -1; y <= 1; y++) {
                            if (x == 0 && y == 0) continue;
                            if (G[(i + x) * sizeCols + (j + y)] >= (higherThreshold * largestG)) {
                                G[i * sizeCols + j] = higherThreshold * largestG;
                                changes = true;
                                breakLoop = true;
                                break;
                            }
                        }
                        if (breakLoop) break;
                    }
                }
                pixelsCanny[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
            }
        }
    } while (changes);
    
    delete[] G;
    pthread_mutex_destroy(&mutex);
    
    return pixelsCanny;
}

// ============================================================================
// PARALLEL CANNY EDGE DETECTION - MAIN FUNCTION
// ============================================================================

void cannyEdgeDetection_parallel(std::string readLocation, std::string writeLocation, 
                                  double lowerThreshold, double higherThreshold) {
    if (readLocation == writeLocation) {
        std::cout << "The read file and save file locations cannot be the same.\n";
        return;
    }
    cv::Mat img = cv::imread(readLocation);
    if (img.empty()) {
        std::cout << "Error: Could not read image from " << readLocation << "\n";
        return;
    }

    uint8_t* pixelPtr = (uint8_t*)img.data;
    int sizeRows = img.rows;
    int sizeCols = img.cols;
    int sizeDepth = img.channels();
    std::vector<int> pixels = imgToArray(img, pixelPtr, sizeRows, sizeCols, sizeDepth);

    // Gaussian blur - parallel
    std::vector<std::vector<double>> kernel = {{2.0, 4.0, 5.0, 4.0, 2.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {5.0, 12.0, 15.0, 12.0, 5.0},
                                               {4.0, 9.0, 12.0, 9.0, 4.0},
                                               {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);
    std::vector<int> pixelsBlur = gaussianBlur_parallel(pixels, kernel, kernelConst, sizeRows, sizeCols, sizeDepth);

    // RGB to Grayscale - parallel
    std::vector<int> pixelsGray = rgbToGrayscale_parallel(pixelsBlur, sizeRows, sizeCols, sizeDepth);

    // Canny filter - parallel
    std::vector<int> pixelsCanny = cannyFilter_parallel(pixelsGray, sizeRows, sizeCols, 1, lowerThreshold, higherThreshold);

    // Write output
    cv::Mat imgGrayscale(sizeRows, sizeCols, CV_8UC1, cv::Scalar(0));
    uint8_t* pixelPtrGray = (uint8_t*)imgGrayscale.data;
    arrayToImg(pixelsCanny, pixelPtrGray, sizeRows, sizeCols, 1);

    cv::imwrite(writeLocation, imgGrayscale);
}
