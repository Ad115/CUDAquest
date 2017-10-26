
__global__  void  AplusB( int *sum,  int *a,  int *b, int n) {
/* 
 * Return the sum of the `a` and `b` arrays
 */
    // Fetch the index
    int i = threadIdx.x;
    // Perform the sum
    ret[i] = a[i] + b[i];
    
} // ---


int main() {
/*
 * Calculate the sum of two vectors using non-managed memory
 */
    int n = 1000;
    
    // <-- HOST memory management
    
        // Create the vectors in the HOST
        int *sum, *a, *b;
        
        sum = malloc( n * sizeof(int) );
        a = malloc( n * sizeof(int) );
        b = malloc( n * sizeof(int) );
        
        // Fill the vectors in the host
        for( int i=0; i<n; i++) {
            
            a[i] = i*i + i;
            b[i] = -i*i; // a[i]+b[i] = i
        }
        
    // <-- DEVICE memory management
    
        // Create the vectors in the DEVICE
        int *d_sum, *d_a, *d_b;
        
        cudaMalloc(&d_sum, n * sizeof(int));
        cudaMalloc(&d_a, n * sizeof(int));
        cudaMalloc(&d_b, n * sizeof(int));
        
        // Copy the vectors to the DEVICE
        cudaMemcpy( d_a, a, n * sizeof(int), cudaMemcpyHostToDevice );
        cudaMemcpy( d_b, b, n * sizeof(int), cudaMemcpyHostToDevice );
        
    // <-- Main calculation
    
        AplusB<<< n, 1 >>>(d_sum, d_a, d_b, n);
        
        // Get the sum vector from the device
        cudaMemcpy(sum, d_sum, n * sizeof(int), cudaMemcpyDeviceToHost);
        
        //Free unneeded memory
        cudaFree(d_sum);
        cudaFree(d_a);
        cudaFree(d_b);
        
    
    // <-- Display results
        
        // Display the results
        for(int i=0; i<n; i++) {
            printf("%d: %d + %d = %d\n", i, a[i], b[i], sum[i]); 
        }
        
        // Free unneeded memory
        free(sum);
        free(a);
        free(b);
        
    return  0;
    
} // ---
