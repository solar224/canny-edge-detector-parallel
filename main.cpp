#include <iostream>
#include <cstdlib>

#include "canny.h"
#include "canny_parallel.h"

int main(int argc, char* argv[]) {
    std::string readLocation = "../images/Sukuna.jpg";
    std::string writeLocation = "../images/SukunaCanny.jpg";
    double lowerThreshold = 0.03;
    double higherThreshold = 0.1;
    
    int numThreads = 1;
    if (argc > 1) {
        numThreads = std::atoi(argv[1]);
        if (numThreads < 1) numThreads = 1;
        if (numThreads > 16) numThreads = 16;
    }
    
    if (argc > 2) {
        readLocation = argv[2];
    }
    
    if (argc > 3) {
        writeLocation = argv[3];
    }
    
    std::cout << "Running Canny Edge Detection with " << numThreads << " thread(s)...\n";
    std::cout << "Input:  " << readLocation << "\n";
    std::cout << "Output: " << writeLocation << "\n";
    
    setNumThreads(numThreads);
    cannyEdgeDetection_parallel(readLocation, writeLocation, lowerThreshold, higherThreshold);
    
    std::cout << "Done!\n";
    
    return 0;
}
