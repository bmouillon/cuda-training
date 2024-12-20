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

__global__ void mykernel(float* r, const float* d, int n, int nn) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    const float* t = d + nn * nn;

    float v[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            v[ib][jb] = HUGE_VALF;
        }
    }
    for (int k = 0; k < n; ++k) {
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ++ib) {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = t[nn * k + i];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = d[nn * k + j];
        }
        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                v[ib][jb] = min(v[ib][jb], x[ib] + y[jb]);
            }
        }
    }
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < n && j < n) {
                r[n * i + j] = v[ib][jb];
            }
        }
    }
}

__global__ void myppkernel(const float* r, float* d, int n, int nn) {
    int ja = threadIdx.x;
    int i = blockIdx.y;

    float* t = d + nn * nn;

    for (int jb = 0; jb < nn; jb += 64) {
        int j = jb + ja;
        float v = (i < n && j < n) ? r[n * i + j] : HUGE_VALF;
        d[nn * i + j] = v;
        t[nn * j + i] = v;
    }
}

static inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

void step(float* r, const float* d, int n) {
    int nn = roundup(n, 64);

    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, 2 * nn * nn * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(rGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nn);
        myppkernel << <dimGrid, dimBlock >> > (rGPU, dGPU, n, nn);
        CHECK(cudaGetLastError());
    }

    // Run kernel
    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(nn / 64, nn / 64);
        mykernel << <dimGrid, dimBlock >> > (rGPU, dGPU, n, nn);
        CHECK(cudaGetLastError());
    }

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