#ifndef ARGS_PARSE_H
#define ARGS_PARSE_H

/*
Argument parsing utilities
--------------------------

Functions for almost pain-free parsing of command line arguments.
Command line arguments are of the form `--key=value` or `-key value`
*/

#include <string.h>
#include "dynamic_str.h"

#define DEFAULT_BUFFER_SIZE 10


char *getKeyFrom(char *str) {
/*
 * From the string `str` of the form "--key=value" returns "key".
 * The key is returned as a dinamically allocated string.
 */   
    // Search the first non-dash (-) character
    for (int i=0; str[i] != '\0'; i++) {
        if (str[i] != '-') {
            str = (str+i);
            break;
        }
    }
    
    // Save the characters until the equals (=) sign
    return readCharsUntil(str, '=');
    
} // --- getKeyFrom



char *getValueFrom(char *str) {
/*
 * From the string `str` of the form "--key=value" obtains "value".
 * The value is returned as a dinamically allocated string.
 */
    // Search for the equals (=) sign
    str = seekCharsUntil(str, '=');
    
    // Save the value string
    return readCharsUntil( str, '\0');
    
} // --- getValueFrom



char *getArgValue(char ***args, char *arg_name) {
/* Searches `àrg_name` as key in the args string array.
 * `args` is supossed to be a list that contains key-vaule pairs of the options
 * read from the command line. Returns the value associated with the key.
 * If the argument or the key is not found, returns NULL.
 */
   for (int i=0; args[i] != NULL; i++) {

       // If the key was found...
       if( strcmp(args[i][0], arg_name)==0 ) {

           return (args[i][1]) ? readCharsFrom(args[i][1]) : NULL; 
           // Return a copy of it's value if there is a value
       }
   }
   // The key was not found
   return NULL;
   
} // --- getArgValue


int searchArg(char ***args, char *arg_name) {
/* Searches `àrg_name` as key in the args string array.
 * `args` is supossed to be a list that contains key-vaule pairs of the options
 * read from the command line. Returns the value associated with the key.
 * If the argument or the key is not found, returns NULL.
 */
   for (int i=0; args[i] != NULL; i++) {

       // If the key was found...
       if( strcmp(args[i][0], arg_name)==0 ) {

           return 1; // Return it's value
       }
   }
   // The key was not found
   return 0;
   
} // --- searchArg



char *findInArguments(char ***args, char *pattern) {
/*
 * Find an argument key that matches pattern. Pattern is a 
 * string with one or more pipe chars, of the form "pattern1|pattern2|patt...". 
 * The arguments in char are searched for a key that matches any "patternX".
 * The value of the key is returned.
 */
    char *value;
    
    do {
        // Fetch the argument
        char *arg = readCharsUntil(pattern, '|');
        // Move to the next argument
        pattern = seekCharsUntil(pattern, '|');
        
        if (*arg == '\0') {
            // The pattern was exhausted and no argument was read
            free(arg);
            // Return a NULL pointer
            value = NULL;
            break;
        }
        
        // Search the argument
        if ( searchArg(args, arg) ) {
            // The argument was found, fetch the value and break the loop
            value = getArgValue(args, arg);
            free(arg);
            break;
            
        } 
        
        // The argument was not found, free the arg used
        free(arg);
            
    } while (1);
    
    return value;
}



 char ***parseArgs(const int argc, char *argv[]) {
/*
 * From the string list `argv` with `argc` entries, get a list of key-value 
 * pairs, one per argument.
 * The arguments are in the form `-[-]key=value` or `-[-]key value`.
 * Returns a list `args` of pairs of strings with the format: args[i][0] = "key", args[i][1] = "value"
 * args[i] = NULL marks the end of the list.
 */

    // Create space for the arguments list (a list of pairs of strings)
    int nargs = 0;
    char ***args = malloc( (nargs+1) * sizeof(*args) ); // Count the NULL at the end
    
    // Parse the arguments
    for (int i=0; i<argc; i++)
    {
        // New argument
        // Allocate space for the key-value pair
        nargs++;
        args = realloc(args, (nargs+1) * sizeof(*args)); // Count the NULL at the end 
        args[nargs-1]= malloc(2 * sizeof(**args));
        
        // Parse first part of the argument
        args[nargs-1][0] = getKeyFrom(argv[i]);
        
        // Check if the string contains an equals sign
        if ( strchr(argv[i], '=') ) { 
            // Assume argument in the format: "-[-]key=value"
            args[nargs-1][1] = getValueFrom(argv[i]);

        } else {
            // Assume argument in the format: "-[-]key value"
            // Check if there is a corresponding value
            if ( (i+1 < argc) && !(strchr(argv[i+1], '-')) ) {
                // Save the value and skim index to the next argument
                args[nargs-1][1] = readCharsUntil(argv[i+1], '\0');
                i++;
            } else {
                // The argument is a loner
                // Set the value as non-existent
                args[nargs-1][1] = NULL;
            }
        }
    }
    // Finally, add an empty entry marking the end of the list
    args[nargs] = NULL;

    return args;
    
} // --- parseArgs


void freeArgs(char ***args)
/*
 * Free the space allocated for the arguments parsed with the parseArgs function.
 */
{
    for(int i=0; args[i] != NULL; i++) {
        // Free the contents of the key
        if (args[i][0]) { free(args[i][0]); }
        // Free the value
        if (args[i][1]) { free(args[i][1]); }
        // Free the key-value pair space
        free(args[i]);
    }
    // Free the array
    free(args);
    
} // --- freeArgs


void printArgs(char ***args)
/*
 * Print the key-value pairs in args (asumming it was created by parseArgs)
 */
{
    printf("Arguments parsed:\n");
    
    if(args) {
        // Print the pairs
        for(int i=0; args[i] != NULL; i++) {
            char *key = args[i][0];
            char *value = args[i][1];
            
            // Print the contents of the key
            if (key) { 
                printf("%s", key);
            }
            printf("  :  ");
            
            // Print the value
            if (value) { 
                printf("%s", value); 
            } else {
                printf("(None)"); 
            }
            printf("\n");
        }
    } else {
        // There are no arguments
        printf("None");
    }
    printf("\n");
    
} // --- printArgs


#undef DEFAULT_BUFFER_SIZE

#endif