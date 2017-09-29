/*
Energy profile
--------------
    
Get the energy profile from a given FASTA sequence.
The energy profile is...BlaBlaBla
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WINDOW_SIZE 5

// Function declarations
int seekToNext(FILE *file, char query);
char *getLineFrom(FILE *file);
char *getFASTASequence(char *filename);
int letterEnergy(char letter);
void calculateEnergyProfile(char *sequence, 
                            int windowSize, 
                            int *saveItHere);
void printEnergyProfile(int *energyProfile, int profileLength);

// Globals
int N_LETTERS = 4;
char *LETTERS = "ACGT";
int LETTER_ENERGIES[] = { -1, 1, 1, -1 };



/* --- << Main function >> --- */


int main() {
    
    // STEP 1: ---> Read the sequence from the FASTA file
    
    char *sequence = getFASTASequence("sequence.fasta");
    
    // Make sure the sequence was read
    printf("Sequence read: %s\n", sequence);
    
    
   /* // STEP 2: ---> Calculate the energy profile of the sequence
    
    int profileLength = sequenceLength - WINDOW_SIZE + 1;
    int energyProfile[ profileLength ];
    
    calculateEnergyProfile(sequence, sequenceLength, WINDOW_SIZE, energyProfile);
    
    // Print the calculated energy profile
    printEnergyProfile( energyProfile, profileLength );*/
    
    
} // --- main



/* --- << Functions >> --- */



int seekToNext(FILE *file, char query) {
/* Move the file after the location of the next matching character.
 * Returns 1 on success, if the character is not found returns 0 and
 * the file is left at EOF.
 */
    char c;
    while( !feof(file) ) {
        // Read a single character from the line
        c = fgetc(file);
        
        if( c == query ) {
            return 1;
        }
    }
    return 0;
    
} // --- seekToNext


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --
// Subsection to implement functionality of <<< getFASTASequence >>>


#define BUFFER_DEFAULT_SIZE 100


typedef struct _Buffer {
/* A structure intended for use as a dynamic string.
 */
    char *content;   // The string per se
    int capacity;// The capacity of the buffer
    int ocupancy;// The number of characters in the buffer
    
} Buffer;


Buffer *newBuffer() {
/* Create a new dynamically allocated Buffer with
 * capacity `capacity` and uninitialized.
 */
    int bufferCapacity = BUFFER_DEFAULT_SIZE;
    
    // Create the structure
    Buffer *buffer = (Buffer *) malloc( sizeof(buffer) );
    // Create the inner buffer
    buffer->content = (char *) malloc( bufferCapacity * sizeof(char) );
    // Initialize
    buffer->capacity = bufferCapacity;
    buffer->ocupancy = 0;
    
    return buffer;
}

void deleteBuffer( Buffer *buffer ) {
/* Free the space occupied by the buffer
 */
    free(buffer->content);
    free(buffer);
    return;
}


void appendToBuffer(char c, Buffer *buffer) {
/* Adds the character `c` to the dynamic buffer `buffer`.
 * If there is no space,the buffer is expanded by `BUFFER_SIZE`/2.
*/
    int charsInBuffer = buffer->ocupancy;
    int bufferCapacity = buffer->capacity;

    // Check if reallocation is needed
    if ( !(charsInBuffer+1 < bufferCapacity) ) {
        // A reallocation is needed
        int newSize = bufferCapacity + BUFFER_DEFAULT_SIZE/2;
        // Reallocate inner string
        buffer->content = (char *) realloc(buffer->content, newSize);
        buffer->capacity = newSize;
    }
        
    // Nope, add it and the terminator
    buffer->content[ charsInBuffer ] = c;
    buffer->content[charsInBuffer + 1] = '\0';
    buffer->ocupancy += 1;
        
    return;
    
} // --- appendToBuffer



char *getLineFrom(FILE *file) {
/* Read one line from the file and return a dynamically allocated buffer with the line.
 * The null terminator is guaranteed to be at the end.
 * Stops reading before EOF or newline character.
*/
    int charactersCount = 0; // Total read characters (not counts null terminator)
    
    //Allocate space for the line
    Buffer *lineBuffer = newBuffer();
    
    char c;
    while( !feof(file) ) {
        // Read a single char from the file
        c = fgetc(file);
        
        // Check newline
        if ( c == '\n') {
            // Finished, get out of the loop
            break;
            
        } else {
            // Save to buffer
            appendToBuffer(c, lineBuffer);
        }
    }
    // Add string terminator
    appendToBuffer('\0', lineBuffer);
    
    // Isolate the inner string and free the structure
    char *line = lineBuffer->content;
    free(lineBuffer);
    
    return line;
    
} // --- getLineFrom



char *getFASTASequence(char *filename) {
/* Read first FASTA sequence from file.
 * Returns a dinamically allocated buffer with the sequence.
 * Returns a NULL buffer if no sequence or header is present.
*/
    char *sequence;
    
    // Open file
    FILE *file = fopen(filename, "r");
        
        // Read file
        while( ! feof(file) ) {
            
            // Go to the next header line
            seekToNext(file, '>');
            
            // Discard the header line
            seekToNext(file, '\n');
            
            // The next line must be the sequence
            sequence = getLineFrom(file);
            
            // Exit the loop & return
            break;
        }
    fclose(file);
    
    return sequence; // Guaranteed to exist.
} // --- getFASTASequence


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


int letterEnergy(char letter) {
/* Uses the globals 'N_LETTERS', 'LETTERS' and 'LETTER_ENERGIES' 
 * to find the energy value corresponding to a given letter.
 */
    for(int i=0; i<N_LETTERS; i++) {
        
        if( LETTERS[i] == letter ) {
            return LETTER_ENERGIES[i];
        }    
    }
    
    // Letter was not found, return a default value
    return 0;
    
} // --- letterEnergy



void calculateEnergyProfile(char *sequence, 
                            int windowSize, 
                            int *saveItHere) {
/* Calculate the energy profile of the sequence.
 * Save into `saveItHere`, assumes is of an appropriate length.
*/
    int sequenceLength = strlen(sequence);
    int totalWindows = 1 + (sequenceLength - windowSize);
    
    // Calculate each window's energy value
    for (int window=0; window<totalWindows; window++) {

        int windowEnergy = 0;
        
        // Go through the window
        for(int i=0; i<windowSize; i++) {
            windowEnergy += letterEnergy( sequence[window + i] );
        }
        // Save the window energy
        saveItHere[window] = windowEnergy;
        
    }
    
    return;
    
} // --- calculateEnergyProfile

void printEnergyProfile(int *energyProfile, int profileLength) {
// Prints the values stored in the energy profile
    
    for( int i=0; i<profileLength; i++) {
        printf("%d ", energyProfile[i]);
    }
    
    printf("\n");
}