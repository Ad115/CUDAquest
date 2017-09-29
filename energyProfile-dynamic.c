/*
Energy profile
--------------
    
Get the energy profile from a given FASTA sequence.
The energy profile is...BlaBlaBla
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WINDOW_SIZE 20

// Function declarations
char *getFASTASequence(char *filename);
int letterEnergy(char letter);
int *getEnergyProfileOf(char *sequence, int windowSize);
void printEnergyProfile(int *energyProfile, char *sequence, int windowSize);

// Globals
int N_LETTERS = 4;
char *LETTERS = "ACGT";
int LETTER_ENERGIES[] = { -1, 1, 1, -1 };



/* --- << Main function >> --- */


int main() {
    
    // STEP 1: ---> Read the sequence from the FASTA file
    
    char *sequence = getFASTASequence("sequence.fasta");
    
    
    // STEP 2: ---> Calculate the energy profile of the sequence
    
    int *energyProfile = getEnergyProfileOf(sequence, WINDOW_SIZE);
    
    // Print the calculated energy profile
    printEnergyProfile( energyProfile, sequence, WINDOW_SIZE );
   

    // END: ---> Free dynamically allocated memory
    
    free(energyProfile);
    free(sequence);
    
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


#define BUFFER_DEFAULT_SIZE 100

char *readUntilNext(FILE *file, char end) {
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
                int newSize = bufferCapacity + BUFFER_DEFAULT_SIZE/2;
                // Reallocate inner string
                buffer = (char *) realloc(buffer, newSize);
                bufferCapacity = newSize;
            }
                
            // Append the character and the terminator
            buffer[ charactersCount ] = c;
            buffer[charactersCount + 1] = '\0';
            charactersCount += 1;
        }
    }
    // Redundant (not when charactersCount = 0)
    buffer[charactersCount] = '\0';
    
    // Free the allocated but unneeded space
    buffer = (char *) realloc(buffer, charactersCount+1);

    return buffer;
    
} // --- readUntilNext


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


char *getLineFrom(FILE *file) {
/* Read one line from the file and return a dynamically allocated buffer with the line.
 * The null terminator is guaranteed to be at the end.
 * Stops reading before EOF or newline character.
*/
    return readUntilNext(file, '\n');
} // --- getLineFrom


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


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
            sequence = readUntilNext(file, '>');
            
            // Exit the loop & return
            break;
        }
    fclose(file);
    
    return sequence; // Guaranteed to exist (worst case: empty string)
    
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


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


int *getEnergyProfileOf(char *sequence, int windowSize) {
/* Calculate the energy profile of the sequence.
 * Returns a dynamically allocated array of energy values 
*/
    int sequenceLength = strlen(sequence);
    int totalWindows = 1 + (sequenceLength - windowSize);
    
    // Make space for the values (1 per window)
    int *energyProfile = (int *) malloc( totalWindows * sizeof(*energyProfile) );
    
    // Calculate each window's energy value
    for (int window=0; window<totalWindows; window++) {

        int windowEnergy = 0;
        
        // Go through the window
        for(int i=0; i<windowSize; i++) {
            windowEnergy += letterEnergy( sequence[window + i] );
        }
        // Save the window energy
        energyProfile[window] = windowEnergy;
        
    }
    
    return energyProfile;
    
} // --- getEnergyProfileOf


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


void printEnergyProfile(int *energyProfile, char *sequence, int windowSize) {
// Prints the values stored in the energy profile
    
    int sequenceLength = strlen(sequence);
    int totalValues = 1 + (sequenceLength - windowSize);
    
    // Print the sequence
    printf("Sequence: %s\n", sequence);
    
    // Print the energy values
    printf("Energy profile:");
    for( int i=0; i<totalValues; i++) {
        printf("%d ", energyProfile[i]);
    }
    
    printf("\n");

} // --- printEnergyProfile