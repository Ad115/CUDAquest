/*
Energy profile
--------------
    
Get the energy profile from a given FASTA sequence.
The energy profile is...BlaBlaBla
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "args_parse.h"
#include "dynamic_str.h"

#define WINDOW_SIZE 20

// Function declarations
int fileExists(char *filename);
char *getFASTASequence(char *filename);
float *getEnergyProfileOf(char *sequence, int windowSize);
void fprintEnergyProfile(FILE *out, float *energyProfile, char *sequence, int windowSize);


// Globals
int N_LETTERS = 4;
char *LETTERS = "ACGT";
float LETTER_ENERGIES[] = { 
    //     A        C        G        T 
        -7.0786, -7.6601, -7.3786, -6.4119,   // A
        -7.5712, -7.2637, -8.8972, -3.3786,   // C
        -7.023,  -9.5936, -7.5712, -7.6601,   // G
        -6.4452, -7.023,  -7.2637, -7.0786    // T
    };


/* --- << Main function >> --- */


int main(int argc, char *argv[]) {
    
    // 1. ---> Parse the arguments to find the input and output file
    
    char ***args = parseArgs(argc, argv);
    
    // Find the input file
    char *input = findInArguments(args, "i|in");
    // Find the output file
    char *output = findInArguments(args, "o|out");
    
    // ---> Calculate the energy profile of the sequence
    
    if ( fileExists(input) ) { // Validate files
        
        char *sequence = getFASTASequence("sequence.fasta");
        
        float *energyProfile = getEnergyProfileOf(sequence, WINDOW_SIZE);
        
        // Open the output file
        FILE *out;
        if ( !output || *output=='\0') {
            // Not a valid file
            // Output to STDOUT
            fprintf(stderr, "Output file not valid");
            out = stdout;
        } else {
			out = fopen(output, "w");
		}
        
        // Print the calculated energy profile
        fprintEnergyProfile( out, energyProfile, sequence, WINDOW_SIZE );
        
        // END: ---> Free dynamically allocated memory
    
        free(energyProfile);
        free(sequence);
    }
    
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
                bufferCapacity += BUFFER_DEFAULT_SIZE/2;
                buffer = (char *) realloc(buffer, bufferCapacity);
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



char *stripFrom(char *str, char unwanted) {
/* Eliminates each occurrence of the character.
 * Reallocates the string buffer to fit.
*/
    int charsStripped = 0;
    int l = strlen(str);
    
    // Pass through the string once and discard the unwanted chars
    char c;
    for(int i=0; i+charsStripped < l+1; i++) {
        // Fetch the next char
        c = str[i+charsStripped];
        
        // Check for the unwanted
        if( c == unwanted ) {
            // We got an unwanted guest: skip it
            charsStripped++;
            i--;
            
        } else if(charsStripped) {
            // Accomodate c in the appropriate position
            str[i] = c;
        }
    }
    // Add the null terminator
    str[ l-charsStripped ] = '\0';
    // Reallocate buffer
    str = (char *) realloc( str, (l-charsStripped+1)*sizeof(char) );
    
    return str;
    
} // --- stripFrom


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


char *getFASTASequence(char *filename) {
/* Read first FASTA sequence from file.
 * Returns a dinamically allocated buffer with the sequence.
 * Returns an empty string if no sequence or header is present.
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
            // Remove newlines
            sequence = stripFrom(sequence, '\n');
            
            // Exit the loop & return
            break;
        }
    fclose(file);
    
    return sequence; // Guaranteed to exist (worst case: empty string)
    
} // --- getFASTASequence


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


float pairEnergy(char a, char b) {
/* Uses the globals 'N_LETTERS', 'LETTERS' and 'LETTER_ENERGIES' 
 * to find the energy value corresponding to a given letter.
 */
	// Find the first letter
	int first=-1;
    for(int i=0; i<N_LETTERS; i++) {
        
        if( LETTERS[i] == a ) {
            first = i;
			break;
        }    
    }
    
    // Find the second letter
	int second=-1;
    for(int i=0; i<N_LETTERS; i++) {
        
        if( LETTERS[i] == b ) {
            second = i;
			break;
        }    
    }
    
    if ( (first != -1) && (second != -1) ) {
		// Fetch the energy value of the pair
		return LETTER_ENERGIES[ N_LETTERS*first + second ];
	}
    
    // Letter was not found, return a default value
    fprintf(stderr, "Letters %c, %c not found\n", a, b);
    return 0;
    
} // --- letterEnergy


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


float *getEnergyProfileOf(char *sequence, int windowSize) {
/* Calculate the energy profile of the sequence.
 * Returns a dynamically allocated array of energy values 
*/
    int sequenceLength = strlen(sequence);
    int totalWindows = 1 + (sequenceLength - windowSize);
    
    // Make space for the values (1 per window)
    float *energyProfile = malloc( totalWindows * sizeof(*energyProfile) );
    
    // Calculate each window's energy value
    for (int window=0; window<totalWindows; window++) {

        float windowEnergy = 0;
        
        // Go through the window
        for(int i=0; i+1<windowSize; i+=2) {
            windowEnergy += pairEnergy( sequence[window+i], sequence[window+i + 1] );
        }
        // Save the window energy
        energyProfile[window] = windowEnergy;
        
    }
    
    return energyProfile;
    
} // --- getEnergyProfileOf


// --- --- - --- --- - --- --- - --- --- - --- --- - --- --- - --- --


void fprintEnergyProfile(FILE *out, float *energyProfile, char *sequence, int windowSize) {
// Prints the values stored in the energy profile
    
    int sequenceLength = strlen(sequence);
    int totalValues = 1 + (sequenceLength - windowSize);
    
    // Print the sequence
    fprintf(out, "Sequence: %s\n", sequence);
    
    // Print the energy values
    fprintf(out, "Energy profile:");
    for( int i=0; i<totalValues; i++) {
        fprintf(out, "%f ", energyProfile[i]);
    }
    
    printf("\n");

} // --- fprintEnergyProfile



int fileExists(char *filename) {
/*
 * Validates that a file exists
 */
    FILE *file = fopen(filename, "r");
    if ( file ) {
        
        fclose(file);
        return 1;
    }
    
    return 0;
    
} // --- fileExists