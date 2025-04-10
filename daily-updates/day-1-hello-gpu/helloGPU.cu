#include <stdio.h>
#include <cuda_runtime.h>

// __global__ indicates this function runs on the GPU.
__global__ void helloFromGPU() {
    if (threadIdx.x == 0) {
        printf("Hello from the GPU!\n");
    }
}

int main() {
    //Launch the kernel with 1 block of 1 thread
    //<<<>>> indicates address of thread
    helloFromGPU<<<1, 1>>>();

    //Synchronize to ensure the GPU finishes before exiting
    //it means kernel will now run on CPU
    cudaDeviceSynchronize();

    printf("Hello from the CPU!\n");
    return 0;
}
