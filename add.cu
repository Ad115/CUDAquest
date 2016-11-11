#include <stdio.h>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    int c = -10;
    int *gpu_c;

    // Create memory on device
    int errorMalloc = cudaMalloc( &gpu_c, sizeof(int) );
    // Call add function on device
    add<<<1,1>>>(2, 3, gpu_c);
    // Copy the result back to the host
    int errorMemcpy = cudaMemcpy( &c, gpu_c, sizeof(int), cudaMemcpyDeviceToHost );
    // Free space used on device
    cudaFree(gpu_c);

    printf("2 + 3 = %d \n", c);
	
	printf("Errors found: cudaMalloc -> %s, cudaMemcpy -> %s)\n", 
		   cudaGetErrorString( errorMalloc ),
		   cudaGetErrorString( errorMemcpy )
	);

    return 0;
}
