#include <iostream>
#include <vector>
#include <random>
#include <chrono>

/**
 * Monte Carlo Pi Estimation
 * This benchmark is excellent for testing GCC optimizations because:
 * 1. It is CPU-bound (not memory-bound).
 * 2. It contains a tight loop that can be vectorized.
 * 3. It relies heavily on math functions and random number generation.
 */

int main() {
    // We use a fixed seed to ensure every run is deterministic 
    // and comparable for the Genetic Algorithm.
    const long long total_points = 10000000; // 10 Million points
    long long points_inside_circle = 0;
    
    unsigned int seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    auto start = std::chrono::high_resolution_clock::now();

    // --- HEAVY COMPUTATION BLOCK ---
    // GCC flags like -funroll-loops and -ffast-math will target this loop
    for (long long i = 0; i < total_points; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        if (x * x + y * y <= 1.0) {
            points_inside_circle++;
        }
    }
    // -------------------------------

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double pi_estimate = 4.0 * points_inside_circle / total_points;

    std::cout << "Estimate: " << pi_estimate << std::endl;
    std::cout << "Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}