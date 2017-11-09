# include <stdio.h>
# include <stdlib.h> // To use the exit function and malloc

/*
 * =============================
 * Find a word in a given string
 * =============================
 * 
 * Usage: find_word <word> <input_file>
 * 
 * Given a word, load the first line of the input file and 
 * search the word in it.
 */

// Global constant
# define NOT_FOUND (-1)

// Function declaration
void validate_arguments(char *argv[], int argc);
FILE *open_or_die(char *filename, char *mode);
char *read_line_from(FILE *file);
int find_word(char *word, char *search_here);



/* --- << Main function >> --- */


int main(int argc, char *argv[]) {
    
    
    // 1. ---> Find the input file and the word to search
    
        // Ensure the arguments where passed correctly
        validate_arguments(argv, argc);
        
        // Get the input file and the word
        char *word = argv[1];
        
        FILE *input = open_or_die(argv[2], "r");
        
            // Get the reference string
            char *search_here = read_line_from(input);
        
        // Close the input file
        fclose(input);
        
        
    // 2. ---> Search the word in the reference string
        
        int found_here = find_word(word, search_here);
    
        
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

void validate_arguments(char *argv[], int argc) {
    /*
     * Check the arguments are OK
     * On failure, exit with error.
     */
    if (argc != 3) {
        fprintf(stderr, "ERROR: Incorrect number of arguments\n");
        fprintf(stderr, "Usage: %s <word> <input_file>\n", argv[0]);
        
        exit(EXIT_FAILURE);
    }
    
} // --- open_or_die


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


FILE *open_or_die(char *filename, char *mode) {
    /*
     * Open the file with the given 'filename' in the given mode.
     * On success, return the file handler.
     * On failure, exit with error.
     */
    FILE *file = fopen(filename, mode);
    
    // Check the file
    if ( !file ) {
        // There was an error opening the file
        exit(EXIT_FAILURE);
    } 
    
    return file;
    
} // --- open_or_die


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


#define BUFFER_DEFAULT_SIZE 100

char *read_until_next(FILE *file, char end) {
    /* Read from the file until `end` or EOF is found.
     * Returns a dynamically allocated buffer with the characters read (excluding `end` and EOF).
     * The null terminator is guaranteed to be at the end.
     * The file is left positioned at the next character after `end` or at EOF.
     */
    
    int charactersCount = 0; // Total read characters (not counts null terminator)
    
    // Allocate space for the contents
    int bufferCapacity = BUFFER_DEFAULT_SIZE;
    char *buffer = (char *) malloc( bufferCapacity * sizeof(char) );
    
    char c;
    while( 1 ) {
        // Read a single char from the file
        c = fgetc(file);
        
        // Check for the character
        if ( c == end || c == EOF ) {
            // Finished, get out of the loop
            break;
            
        } else {
            // Append `c` to the line ---
            
            // Check if a reallocation is needed
            if ( !(charactersCount+1 < bufferCapacity) ) {
                // A reallocation is needed
                bufferCapacity += BUFFER_DEFAULT_SIZE/2;
                buffer = (char *) realloc(buffer, bufferCapacity);
            }
                
            // Append the character and the terminator
            buffer[ charactersCount ] = c;
            buffer[charactersCount + 1] = '\0';
            charactersCount += 1;
        }
    }
    // Not redundant when charactersCount = 0
    buffer[charactersCount] = '\0';
    
    // Free the allocated but unneeded space
    buffer = (char *) realloc(buffer, charactersCount+1);

    return buffer;
    
} // --- read_until_next


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


char *read_line_from(FILE *file) {
    /*
     * Read the next line of the file.
     * Stops at newline or EOF.
     * Returns a malloc'd buffer with the characters read, excluding the newline or EOF.
     * The returned buffer is guaranteed to be properly null-terminated
     */
    
    return read_until_next(file, '\n'); // Read until newline or EOF
    
} // --- open_or_die


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


int find_word(char *word, char *search_here) {
    /*
     * Search for the given word in the search_here string.
     * At first occurrence, returns the starting position.
     * If the word was not found, return NOT_FOUND.
     */
    
    // Iterate through each letter and try to find the word
    for (int i=0; search_here[i] != '\0'; i++) {
        
        // Check if the word is found from here
        for (int j=0; ; j++) {
            
            // When you have reached the end of the word, 
            if (word[j] == '\0') {
                // ...It means you found it!!!
                return i;
            }
            
            // Check if the letters coincide
            if (search_here[i+j] != word[j]) {
                // Not a match, continue at the next position
                break;
            }
        }
        
    }
    
    // Completed the loops, that means the word was not found
    return NOT_FOUND;
    
} // --- find_word
