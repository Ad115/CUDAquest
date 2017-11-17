/*
Energy profile
--------------
    
Get the energy profile from a given FASTA sequence.
The energy profile is...BlaBlaBla
*/

#include <stdio.h>
#include <string.h>

#define MAX_LINE 500
#define WINDOW_SIZE 5

// Function declarations
int readLineFrom(FILE *file, char *saveItHere, int bufferSize);
int readFASTASequence(char *filename, char *saveItHere, int space);
int letterEnergy(char letter);
void calculateEnergyProfile(char *sequence, 
                            int sequenceLength, 
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
    
    char sequence[MAX_LINE+1]; // The sequence has MAX_LINE chars
    
    int sequenceLength = readFASTASequence("Escherichia_coli.fa", 
                                           sequence, 
                                           MAX_LINE+1);
    // Make sure the sequence was read
    printf("Sequence read: %s\n", sequence);
	
    
    // STEP 2: ---> Calculate the energy profile of the sequence
    
    int profileLength = sequenceLength - WINDOW_SIZE + 1;
    int energyProfile[ profileLength ];
    
    calculateEnergyProfile(sequence, sequenceLength, WINDOW_SIZE, energyProfile);
    
    // Print the calculated energy profile
    printEnergyProfile( energyProfile, profileLength );
    
    
} // --- main



/* --- << Functions >> --- */



int readLineFrom(FILE *file, char *saveItHere, int bufferSize) {
/* Read one line from the file and return the number of charactres read.
 * Saves read line into the `saveItHere` buffer and stop at newline, EOF,
 * or when `bufferSize - 1` characters are read.
*/
    int charactersCount = 0; // Total read characters
    
    char c;
    while( !feof(file) && charactersCount < bufferSize-1) {
        // Read a single char from the file
        c = fgetc(file);
        
        // Check newline
        if ( c == '\n') {
            // Add string terminator and return
            saveItHere[charactersCount+1] = '\0';
            return charactersCount;
        } else {
            // Save to buffer
            saveItHere[charactersCount] = c;
            charactersCount++;
        }
    }
    
    return charactersCount;

} // --- readLineFrom



int readFASTASequence(char *filename, char *saveItHere, int bufferSize) {
/* Read first FASTA sequence from file.
 * Read into `saveItHere` at most `bufferSize`-1 chars. 
 * Returns the characters read.
*/
    
    // Open file
    FILE *file = fopen(filename, "r");
    
        // Temporary buffer
        char line[bufferSize]; // Doesn't need to be bigger
        
        while( ! feof(file) ) {
            // Read a line from the file into the buffer
            int lineLength = readLineFrom(file, line, bufferSize);
            
            // Check if you are in the header
            if( line[0] != '>' ) {
                // You are not in the header: save the line
                strncpy(saveItHere, line, bufferSize);
                
                // Make sure the terminator is there
                saveItHere[bufferSize-1] = '\0';
                
                return lineLength; 
            }
            
        }
    fclose(file);
} // --- readFASTASequence



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
                            int sequenceLength, 
                            int windowSize, 
                            int *saveItHere) {
/* Calculate the energy profile of the sequence.
 * Save into `saveItHere`, assumes is of an appropriate length.
*/
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
