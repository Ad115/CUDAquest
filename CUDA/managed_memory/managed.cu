#include <stdio.h>


__global__  void  AplusB( int *sum,  int *a,  int *b, int n) {
/*
 * Return the sum of the `a` and `b` arrays
 */
    // Fetch the index
    int i = blockIdx.x;
    // Perform the sum
    sum[i] = a[i] + b[i];
} // ---


int main() {
/*
 * Calculate the sum of two vectors using managed memory
 */
    int n = 1000;

    // <-- GLOBAL memory management

        // Create the vectors (managed memory)
        int *sum, *a, *b;

        cudaMallocManaged( &sum, n*sizeof(int) );
        cudaMallocManaged( &a, n*sizeof(int) );
        cudaMallocManaged( &b, n*sizeof(int) );

        // Fill the vectors in the host
        for( int i=0; i<n; i++) {

            a[i] = i*i + i;
            b[i] = -i*i; // a[i]+b[i] = i
        }


    // <-- Main calculation

        // Note how we don't copy TO the DEVICE

        AplusB<<< n, 1 >>>(sum, a, b, n);

	// wait for the DEVICE to finish
	cudaDeviceSynchronize();

    // <-- Display results

        // Display the results
        for(int i=0; i<n; i++) {
            // Note how we don't copy FROM the DEVICE
            printf("%d: %d + %d = %d\n", i, a[i], b[i], sum[i]); 
        }

        // Free unneeded memory
        cudaFree(sum);
        cudaFree(a);
        cudaFree(b);

    return 0;

} // ---
