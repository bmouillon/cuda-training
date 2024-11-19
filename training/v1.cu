#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

using namespace std;

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

__global__ void mykernel(float* r, const float* d, int n) {
    // TO DO
}

static inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

void step(float* r, const float* d, int n) {
    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));
    mykernel << <dimGrid, dimBlock >> > (rGPU, dGPU, n);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}

int main() {
    constexpr int n = 20000;
    // Generate a random graph
    vector<float> d(n * n);
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < n * n; ++i) {
        d[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    // Compute shortest 2-edge paths
    vector<float> r(n * n);
    auto start = chrono::high_resolution_clock::now();
    step(r.data(), d.data(), n);
    auto end = chrono::high_resolution_clock::now();
    // // Display results
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << r[i * n + j] << " ";
    //     }
    //     cout << "\n";
    // }
    chrono::duration<float> duration = end - start;
    cout << "Time elapsed: " << duration.count() << " seconds\n";
    return 0;
}