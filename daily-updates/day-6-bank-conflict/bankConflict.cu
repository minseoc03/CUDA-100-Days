#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32
#define DATA_SIZE (TILE_WIDTH * TILE_WIDTH)

// Kernel that intentionally causes bank conflicts
__global__ void bankConflictKernel(float *input, float *output) {
    __shared__ float sharedData[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = ty * TILE_WIDTH + tx;

    // Access pattern causing bank conflicts
    int conflictIndex = tx * 2 % TILE_WIDTH;  // Artificial non-optimal pattern
    sharedData[ty][conflictIndex] = input[index];

    __syncthreads();

    output[index] = sharedData[ty][conflictIndex];
}

// Kernel that avoids bank conflicts by using a contiguous access pattern
__global__ void optimizedKernel(float *input, float *output) {
    __shared__ float sharedData[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = ty * TILE_WIDTH + tx;

    // Coalesced access pattern
    sharedData[ty][tx] = input[index];

    __syncthreads();

    output[index] = sharedData[ty][tx];
}

int main() {
    size_t size = DATA_SIZE * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < DATA_SIZE; i++) {
        h_input[i] = (float)(i);
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(1, 1);

    cudaEvent_t start, stop;
    float timeConflict, timeOptimized;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Warm-Up for Bank Conflict Kernel ---
    bankConflictKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    cudaDeviceSynchronize();

    // Measure bankConflictKernel execution time
    cudaEventRecord(start);
    bankConflictKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeConflict, start, stop);

    // --- Warm-Up for Optimized Kernel ---
    optimizedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    cudaDeviceSynchronize();

    // Measure optimizedKernel execution time
    cudaEventRecord(start);
    optimizedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeOptimized, start, stop);

    printf("Bank Conflict Kernel Time:    %f ms\n", timeConflict);
    printf("Optimized Kernel Time:        %f ms\n", timeOptimized);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
