#include <stdio.h>

__global__ void AplusB(int *ret, int a, int b) {
/*
* Simple unimportant kernel
*/
    ret[threadIdx.x] = a + b + threadIdx.x;
}


int main() {

    // Create a managed space
    int *ret;
    cudaMallocManaged(&ret, 1000 * sizeof(int));

    // Call the kernel 
    AplusB<<< 1, 1000 >>>(ret, 10, 100);
        cudaDeviceSynchronize();

    // Print the results
    for(int i=0; i<1000; i++) {
        printf("%d: A+B = %d\n", i, ret[i]);
    }

    // Free the unneeded memory
    cudaFree(ret); 

    return  0;
}


