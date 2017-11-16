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
void __global__ find_word_kernel(char *word, char *search_here, int *found_here, int ref_length) {
    /*
     * Search for the given word in the search_here string.
     * At first occurrence, returns the starting position.
     * If the word was not found, return -1.
     */
    
    // The starting position of each thread
    int start = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    if (start < ref_length-1) { // Check for a valid position
        
        //printf("Process starting from position %d\n\tword: %s\n\tstring: %s\n", start, word, search_here);
        int found = 1; // Pretend you found it
        int letters_coincide;
    
        // ---> Check if the word is found from here
        
        for (int j=0; word[j] != '\0'; j++) {
            // Check if the letters coincide
            letters_coincide = (search_here[start+j] == word[j]);
            found = (found && letters_coincide);
        }
        
        // Place your mark
        found_here[start] = found;
        
    }
    
    return;
} // --- find_word_kernel


// ----------------------------------------------------------------------------


/* --- << Main function >> --- */


int main(int argc, char *argv[]) {
    
    // 1. ---> Find the input file and the word to search
    
        char *search_here = argv[1];
        char *word = argv[2]
        
        
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
        
        
    // 4. ---> Cleanup
    
        free(search_here);
        
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
        
        // Prepare for the arrival of the result
        int *found_here_tmp;
        cudaMallocManaged(&found_here_tmp, str_length * sizeof(int));
        for (int i=0; i < str_length; i++) {
            found_here_tmp[i] = 0;
        }
    
    
    // 2. --- > Prepare and launch the Kernel
    
        // Calculate the total threads to use (one per window)
        int total_threads = (str_length - word_length) + 1;
        
        // Calculate the blocks needed for that
        int blocks = (total_threads + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
        
        printf("Launching %d threads in %d blocks\n", THREADS_PER_BLOCK, blocks);
        
        // Launch the kernel
        find_word_kernel<<<blocks, THREADS_PER_BLOCK>>>(word_tmp, str_tmp, found_here_tmp, str_length);
        
        cudaDeviceSynchronize();
    
        
    // 3. --- > Analyze the result
        for (int i=0; i<ref_length; i++) {
            if ( found_here_tmp[i] ) {
                found_here = i;
                break;
            }
        }
        
    
    // 4. ---> Cleanup and return
    
        // Free unneeded memory
        cudaFree(found_here_tmp);
        cudaFree(word_tmp);
        cudaFree(ref_tmp);
        
    return found_here;
    
} // --- find_word_in_gpu
