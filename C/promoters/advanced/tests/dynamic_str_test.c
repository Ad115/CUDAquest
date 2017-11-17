/*
Dynamic string utilities test
-----------------------------

The dynamic string utilities module "dynamic_str.h" contains functions and 
structures for almost pain-free handling of dynamic char strings.
This is a test suite to ensure everything is OK.
*/

#include <stdio.h>
#include "../dynamic_str.h"

int main() {
	
	char *buffer, *chars;
	String *str;
	char c;

    // --> testing newString
	
	printf("\nTesting newString:\n");
    
    str = newString();
    
    printf("Printing an empty string:\n"
	
           "First way:\t'"
    );
    printString(str);
    
    printf("'\nSecond way:\t'%s'\n", str->content);
    
	buffer = popStringContent(str);
    printf("Third way:\t'%s'\n", buffer);
	free(buffer);
	
	
	// --> testing newStringUntil
	
	printf("\nTesting newStringUntil:\n");
    
	buffer = "Hola k aseze";
	c = 'z';
	printf("Original string:\t%s\n", buffer);
	
    str = newStringUntil(buffer, 'z');
    
    printf("String until %c:\n"
	
           "First way:\t'"
    , c);
    printString(str);
    
    printf("'\nSecond way:\t'%s'\n", str->content);
    
    buffer = popStringContent(str);
    printf("Third way:\t'%s'\n", buffer);
	free(buffer);
	
	
	// --> testing newStringFrom
	
	printf("\nTesting newStringFrom:\n");
    
	buffer = "Hola k aseze";
	printf("Original string:\t%s\n", buffer);
	
    str = newStringFrom(buffer);
    
    printf("String read:\n"
	
           "First way:\t'"
    );
    printString(str);
    
    printf("'\nSecond way:\t'%s'\n", str->content);
    
    buffer = popStringContent(str);
    printf("Third way:\t'%s'\n", buffer);
	free(buffer);
	
	
	// --> testing appendToString
	
	printf("\nTesting appendToString:\n");
    
	buffer = "Hola k aseze";
	c = 'n';
	str = newStringFrom(buffer);
	printf("Original string:\t%s\n", str->content);
	
    appendToString(str, 'n');
    printf("String with added 'n':\n"
	
           "First way:\t'"
    );
    printString(str);
    
    printf("'\nSecond way:\t'%s'\n", str->content);
    
    buffer = popStringContent(str);
    printf("Third way:\t'%s'\n", buffer);
	free(buffer);
	
	// -- -- Second test for appendToString
	
	printf("Second test (brute force append)\n");
	
	str = newString();
	// Append lots of characters
	for (int i=0; i<200; i++) {
		appendToString(str, 'x');
		printf("'%s': %d\n", str->content, (int)strlen(str->content));
	}
	deleteString(str);
	
	// --> testing popStringContent
	// --> testing printString
	// --> testing deleteString
	printf("\nThe function `popStringContent`was already tested with the previous tests\n");
	printf("\nThe function `printString`was already tested with the previous tests\n");
	printf("\nThe function `deleteString`was already tested with the previous tests\n");
	
	
	// --> testing readCharsUntil
	
	printf("\nTesting readCharsUntil:\n");
    
	buffer = "Hola k aseze";
	c = 'z';
	printf("Original string:\t%s\n", buffer);
	
    chars = readCharsUntil(buffer, c);
    
    printf("String until %c:\t'%s'\n", c, chars);
	free(chars);
	
	
	// --> testing readCharsFrom
	
	printf("\nTesting readCharsFrom:\n");
    
	buffer = "Hola k aseze";
	c = 'z';
	printf("Original string:\t%s\n", buffer);
	
    chars = readCharsFrom(buffer);
    
    printf("String read:\t'%s'\n", chars);
	free(chars);
	
	
	// --> testing seekCharsUntil
	
	printf("\nTesting seekCharsUntil:\n");
    
	buffer = "Hola k aseze";
	c = 'z';
	printf("Original string:\t%s\n", buffer);
	
    chars = seekCharsUntil(buffer, c);
    
    printf("String after %c:\t'%s'\n", c, chars);
	
	
	// --> testing seekCharsAfterLast
	
	printf("\nTesting seekCharsAfterLast:\n");
    
	buffer = "askjdslfjlasjlskjljdsawwwwHola k aze";
	c = 'w';
	printf("Original string:\t%s\n", buffer);
	
    chars = seekCharsAfterLast(buffer, c);
    
    printf("String after all %c's:\t'%s'\n", c, chars);
	
}
