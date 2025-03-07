#include <stdio.h>

// A simple NxN row-major matrix multiplication
// extern "C" for name unmangling
extern "C" __global__
void matMulKernel(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float val = 0.0f;
        for (int k = 0; k < N; k++) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}
