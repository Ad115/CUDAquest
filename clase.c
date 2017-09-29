/*
Energy profile
--------------
    
Get the energy profile from a given FASTA sequence.
The energy profile is...BlaBlaBla
*/

#include <stdio.h>
#include <string.h>

#define MAX_LINE 500

// Function declarations
int readLineFrom(FILE *file, char *saveItHere, int bufferSize);
int readFASTASequence(char *filename, char *saveItHere, int space);

/* --- << Main function >> --- */

int main() {
    
    // STEP 1: ---> Read the sequence from the FASTA file
    
    char sequence[MAX_LINE+1]; // The sequence has MAX_LINE chars
    
    int sequenceLength = readFASTASequence("sequence.fasta", 
                                           sequence, 
                                           MAX_LINE+1);
    // Make sure the sequence was read
    printf("Sequence read: %s\n", sequence);    
    
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


