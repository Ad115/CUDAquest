# include <stdio.h>
# include <stdlib.h> // To use the exit function and malloc
# include <string.h>

/*
 * ============================================
 * Find a word in a given string (CUDA version)
 * ============================================
 * 
 * Usage: find_word <word> <input_file>
 * 
 * Given a word, load the first line of the input file and 
 * search the word in it. This version uses a CUDA-enabled
 * graphics card.
 */

// Global constant
# define NOT_FOUND (-1)
# define THREADS_PER_BLOCK (128)

// Function declaration
int find_word_in_gpu(char *word, char *search_here);


// ----------------------------------------------------------------------------


// Kernel definition
void __global__ find_word_kernel(char *word, char *search_here, int ref_length, int *result) {
    /*
     * Search for the given word in the search_here string.
     * 
     * At first occurrence, returns the starting position. If the word was not 
     * found, return NOT_FOUND.
     */
    
    // 1. --- > Prepare for execution
    
        // Allocate shared memory for the result
        __shared__ int found_here[THREADS_PER_BLOCK];
    
        // The starting position of each thread
        int start = (blockDim.x * blockIdx.x) + threadIdx.x;
        
        // The shared memory index for this thread
        int shared_idx = threadIdx.x;
        
    
    // 1. --- > Search for the word
    
        
        
        if (start < ref_length-1) { // Check for a valid position
            
            int found = 1; // Pretend you found it
            int letters_coincide;
        
            // ---> Check if the word is found from here
            
            for (int j=0; word[j] != '\0'; j++) {
                // Check if the letters coincide
                letters_coincide = (search_here[start+j] == word[j]);
                found = (found && letters_coincide);
            }
            
            // ---> Place your mark
            if (found) { 
                // Place position if it was found
                found_here[shared_idx] = start;
                
            } else {
                found_here[shared_idx] = 0;
            }
            
        } else { // Non working thread, initialize shared memory
            // You will definitely NOT find it here
            found_here[start] = 0;
        }
        
        // Wait until everyone finishes
        __syncthreads();
        
    // 2. --- > Reduce the result on every thread
        
    
        // ---> Reduce the results to one per block
        int threads_per_block = blockDim.x;
        int i = (threads_per_block+1)/2;
        printf("Reduction started: thread %d, found_here[here] = %d\n", threadIdx.x, found_here[shared_idx]);
        
        while( i != 0 ) { 
            
            
            // Reduce halving the results on each iteration
            if (threadIdx.x < i) {
                
                // Check if the entries are within reach
                if ( shared_idx + i < threads_per_block ) {
                    printf("Reducing entries %d and %d (%d, %d). Thread %d.\n", shared_idx, shared_idx+i, found_here[shared_idx], found_here[shared_idx+i], threadIdx.x);
                    // Check if it was found here
                    found_here[shared_idx] = (found_here[shared_idx] ? found_here[shared_idx] : found_here[shared_idx+i]);
                }
            }
            
            // Prepare the next reduction
            i/=2;
            __syncthreads();
        }
        
        // ---> Save the block's reduction and return
        
        if (threadIdx.x == 0) {
            printf("Reduced block %d: %d\n", blockIdx.x, found_here[shared_idx]);
            result[blockIdx.x] = found_here[shared_idx];
        }
    
    return;
} // --- find_word_kernel


// ----------------------------------------------------------------------------


/* --- << Main function >> --- */


int main(int argc, char *argv[]) {
    
    // 1. ---> Find the input file and the word to search
    
        char *search_here = argv[1];
        char *word = argv[2];
        
        
    // 2. ---> Search the word in the reference string
        
        int found_here = find_word_in_gpu(word, search_here);
    
        
    // 3. ---> Display the results
        
        if( found_here == NOT_FOUND ) {
            // The word was not found
            printf("Sorry, the word was not found in the reference string\n");
            printf("Word: %s\nReference string: %s\n\n", word, search_here);
            
        } else {
            // The word was found
            printf("The word was found at position: %d\n", found_here);
            
            // Signal the position
            printf("Word: %s\nReference string: %s\n", word, search_here);
            printf("                   ");
            for (int i=0; i < found_here-1; i++)
                printf(" ");
            printf("^\n\n");
        }
        
        
    // 4. ---> Finish!
        
    return 0;
        
} // --- main



// ----------------------------------------------------------------------------

/* --- << Functions >> --- */


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


int find_word_in_gpu(char *word, char *search_here) {
    /*
     * Search for the given word in the search_here string.
     * 
     * At first occurrence, returns the starting position. If the word was not 
     * found, return NOT_FOUND. Uses a CUDA-enabled graphics card.
     */
    
    // 1. --- > Prepare the data in the CPU
        
        // Lookup the lengths of the words
        int word_length = strlen(word);
        int str_length = strlen(search_here);
        int found_here = NOT_FOUND;
        
        // Copy the word to the GPU
        char *word_tmp;
        cudaMallocManaged(&word_tmp, word_length * sizeof(char));
        strcpy(word_tmp, word);
        
        // Copy the search_string to the GPU
        char *str_tmp;
        cudaMallocManaged(&str_tmp, str_length * sizeof(char));
        strcpy(str_tmp, search_here);
    
    
    // 2. --- > Prepare and launch the Kernel
    
        // Calculate the total threads to use (one per window)
        int total_threads = (str_length - word_length) + 1;
        
        // Calculate the blocks needed for that
        int blocks = (total_threads + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
        
        printf("Launching %d threads in %d blocks\n", THREADS_PER_BLOCK, blocks);
        
        // Prepare for the arrival of the results
        int *partial_results;
        cudaMallocManaged(&partial_results, blocks * sizeof(int));
        for (int i=0; i < blocks; i++) {
            partial_results[i] = 0;
        }
        
        // Launch the kernel
        find_word_kernel<<<blocks, THREADS_PER_BLOCK>>>(word_tmp, str_tmp, str_length, partial_results);
        cudaDeviceSynchronize();
    
        
    // 3. --- > Analyze the result
        for (int i=0; i<blocks; i++) {
            if ( partial_results[i] ) {
                found_here = partial_results[i];
                break;
            }
        }
        
    
    // 4. ---> Cleanup and return
    
        // Free unneeded memory
        cudaFree(partial_results);
        cudaFree(word_tmp);
        cudaFree(str_tmp);
        
    return found_here;
    
} // --- find_word_in_gpu
