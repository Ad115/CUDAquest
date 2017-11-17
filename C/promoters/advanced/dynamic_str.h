# ifndef DYNAMIC_STR_H
# define DYNAMIC_STR_H

/*
Dynamic string utilities
------------------------

Functions and structures for almost pain-free handling of dynamic char strings.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define BUFFER_DEFAULT_SIZE 100


typedef struct _String {
/* A structure intended for use as a dynamic string.
 */
    char *content;   // The string per se
    int capacity;// The capacity of the string
    int ocupancy;// The number of characters in the string
    
} String;



//  --->   Function declarations

    // Dynamic string structure methods
    String *newString( void );
    
    String *newStringUntil(char *buffer, char c);
    
    String *newStringFrom(char *buffer);
    
    void deleteString( String *str );
    
    void appendToString(String *str, char c);
    
    char *popStringContent( String *str );
    
    void printString( String *str );


    // Functions to handle character buffers
    char *readCharsUntil(char *buffer, char c);

    char *readCharsFrom(char *buffer);

    char *seekCharsUntil(char *buffer, char c);

    char *seekCharsAfterLast(char *buffer, char c);

//  <--- Function declarations



String *newString( void ) {
/* Create a new dynamically allocated String with
 * capacity `capacity`. The string is empty
 */
    int strCapacity = BUFFER_DEFAULT_SIZE;
    
    // Create the structure
    String *str = malloc( sizeof(str) );
    
    // Create the inner string
    str->content = malloc( strCapacity * sizeof(*str->content) );
    str->content[0] = '\0';
    
    // Initialize the structure
    str->capacity = strCapacity;
    str->ocupancy = 0;
    
    return str;

} // --- newString



String *newStringUntil(char *buffer, char c) {
/* Create a new dynamically allocated String initialized from the contents of
 * the `buffer` before the first occurrence of `c` or '\0'.
 */
    // Create a new dynamical string
    String *str = newString();
    
    // Copy char by char the contents of the buffer
    int i=0;
    while (1) {
        // Get next character
        char next = buffer[i];

        if ((next == c) || (next == '\0')) {
            // Character found, exit the loop
            break;

        } else {
            // Add the character to the string
            appendToString(str, next);
            i++;
        }
    }

    // Return the created object
    return str;

} // --- newStringUntil



String *newStringFrom(char *buffer) {
/* Create a new dynamically allocated String initialized from the contents of
 * the argument.
 */
    // Read the buffer until the string terminator
    return newStringUntil(buffer, '\0');

} // --- newStringUntil



void deleteString( String *str ) {
/* Free the space occupied by the string
 */
    free(str->content);
    free(str);
    return;

} // --- deleteString



void appendToString(String *str, char c) {
/* Adds the character `c` to the dynamic string `str`.
 * If there is no space,the string is expanded by `BUFFER_DEFAULT_SIZE`/2.
*/
    int charsInString = str->ocupancy;
    int strCapacity = str->capacity;

    // Check if reallocation is needed
    int neededSpace = (charsInString+1); // Don't count the '\0'
    
    if ( !(neededSpace < strCapacity) ) {
        // A reallocation is needed
        int newSize = strCapacity + BUFFER_DEFAULT_SIZE/2;
        // Reallocate inner string
        str->content = realloc(str->content, newSize);
        str->capacity = newSize;
    }

    // Add the character and the terminator
    str->content[ charsInString ] = c;
    str->content[charsInString + 1] = '\0';
    str->ocupancy += 1;
        
    return;
    
} // --- appendToString



char *popStringContent( String *str ) {
/* Deallocates the String but keeps the inner char buffer.
 * Returns it after adjusting the space it occupies in memory.
*/
    // Save the contents and get rid of the wrapper structure
    char *buffer = str->content;
    free(str);
    
    return buffer;
    
} // --- popStringContent



void printString( String *str ) {
/*
 * Print the content of the dynamic string structure
 */
    printf("%s", str->content);
    return;
    
} // --- printStr



char *readCharsUntil(char *buffer, char c) {
/*
 * Utility function that copies the contents of `buffer` to another dynamically
 * allocated one. Stops copying before the first `c` or '\0'. Adjsts the space
 * allocated to the minimal necessary and adds a null terminator at the end.
 */
    // Create a new dynamical string with the contents of the buffer up to
    // before the first appearance of `c`
    String *str = newStringUntil(buffer, c);
    
    // Return the structure contents and free the unneeded memory
    return popStringContent( str );
    
} // --- readCharsUntil



char *readCharsFrom(char *buffer) {
/*
 * Utility function that copies the contents of `buffer` to another dynamically
 * allocated one. Adjusts the space allocated to the minimal necessary and adds a null terminator at the end.
 */
    // Create a new dynamical string with the contents of the buffer up to
    // before the first appearance of `c`
    String *str = newStringFrom(buffer);
    
    // Return the structure contents and free the unneeded memory
    return popStringContent( str );
    
} // --- readChars




char *seekCharsUntil(char *buffer, char c) {
/*
 * Utility function that searches `c` in the buffer and returns a pointer to 
 * the next position. If the character is not in the string, return a pointer
 * to the string terminator.
 */
    
    // Search for `c` or the string terminator
    int i=0;
    while (1) {
        // Get next character
        char next = buffer[i];
        
        if (next == '\0') {
            // Character not found, exit
            break;
            
        } else if (next == c) {
            // Character found.
            // Skip the character and exit the loop
            i++;
            break;
        }
        
        i++;
    }
    
    // Return a pointer to the next character
    return (buffer + i);
    
} // --- seekCharsUntil



char *seekCharsAfterLast(char *buffer, char c) {
/*
 * Utility function that searches contiguous occurrences of `c` in the buffer
 * and returns a pointer to the next position. If the character is not in the 
 * string, return a pointer to the string terminator.
 */
    // Find the first occurrence of `c`
    buffer = seekCharsUntil(buffer, c);
    
    // Search for a char different from `c` or the string terminator
    int i=0;
    while (1) {
        // Get next character
        char next = buffer[i];
        
        if ( (next != c) || (next == '\0') ) {
            // A different character was found.
            // Exit the loop
            break;
            
        }
        
        i++;
    }
    
    // Return a pointer to the next character
    return (buffer+i);
    
} // --- seekCharsAfterLast


# endif