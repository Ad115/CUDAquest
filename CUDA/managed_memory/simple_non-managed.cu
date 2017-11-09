#include <stdio.h>

__global__  void  AplusB( int  *ret,  int  a,  int  b) {
/*
 * Simple unimportant kernel
 */
    ret[threadIdx.x] = a + b + threadIdx.x;
}


int main() {

    // Create space in the device
    int *ret;
    cudaMalloc(&ret, 1000 * sizeof(int));

    // Call the kernel
    AplusB<<< 1, 1000 >>>(ret, 10, 100);

    // Recieve the results
    int *host_ret = (int *)malloc(1000 * sizeof(int));
    cudaMemcpy(host_ret, ret, 1000 * sizeof(int), cudaMemcpyDefault);

    // Display the results
    for(int i=0; i<1000; i++) {
        printf("%d: A+B = %d\n", i, host_ret[i]); 
    }

    // Free unneeded memory
    free(host_ret);
    cudaFree(ret); 

    return  0;
}
