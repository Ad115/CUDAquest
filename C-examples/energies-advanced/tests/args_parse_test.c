/*
Argument parsing utilities test
--------------------------------------------

The argument parsing utilities module "args_parse.h.h" contains functions and 
structures for almost pain-free handling of dynamic char strings. This is a 
test suite to ensure everything is OK.
*/

#include <stdio.h>
#include "../args_parse.h"

void checkArgument(char ***args, char *arg);
void checkPattern(char ***args, char *pattern);

int main(int argc, char *argv[]) {

    // --> testing parseArgs
	
	printf("\nTesting parseArgs:\n");
	
	printf("Original arguments (%d):\n", argc);
	for (int i=0; i<argc; i++) {
		printf("%d: '%s'\n", i, argv[i]);
	}

    char ***args = parseArgs(argc, argv);

    printArgs(args);
	
	
	// --> testing searchArgs
	
	printf("Testing searchArgs:\n");
	
	checkArgument(args, "something");
	checkArgument(args, "sleeping");
	checkArgument(args, "other");
	
	printf("\n");
	
	
	// --> testing findInArguments
	
	printf("Testing findInArguments:\n");
	
	checkPattern(args, "o|out");
	checkPattern(args, "i|in");
	checkPattern(args, "other|else");
	
	printf("\n");
	
	freeArgs(args);
	
	
	// ---> testing getKeyFrom
	// ---> testing getValueFrom
	// ---> testing getArgValue
	// ---> testing freeArgs
	// ---> testing printArgs
	printf("\nThe function `getValueFrom`was already tested with the previous tests\n");
	printf("\nThe function `getKeyFrom`was already tested with the previous tests\n");
	printf("\nThe function `getArgValue`was already tested with the previous tests\n");
	printf("\nThe function `freeArgs` was already tested with the previous tests\n");
	printf("\nThe function `printArgs`was already tested with the previous tests\n\n");
	
	
} //--- main



// ---> Functions <---

void checkArgument(char ***args, char *arg) {
/*
 * Prints whether the argument was found or not
 */
	if ( searchArg(args, arg) ) {
		
		char *value = getArgValue(args, arg);
		
		if (value) {
			printf("Argument %s was passed, with value %s\n", arg, value);
			free(value);
			
		} else {
			printf("Argument %s was passed, with no value\n", arg);
		}
	} else {
		printf("Argument %s was not passed\n", arg);
	}
	
	return;
}

void checkPattern(char ***args, char *pattern) {
/*
 * Prints whether the argument was found or not using findInArguments
 */
	char *value = findInArguments(args, pattern);
	
	if ( value ) {
			printf("Argument matching %s was passed, with value %s\n", pattern, value);
			free(value);
	} else {
		printf("Argument matching %s was not passed or passed with no value\n", pattern);
	}
	
	return;
}
