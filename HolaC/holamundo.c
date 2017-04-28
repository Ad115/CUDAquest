/*
 Esto está comentado
 
 
 
 y esto tmb xD
 
*/

// Regresa 10
#include <stdio.h>


float suma(float a, float b){
	
	return a+b;
}

int main() {
	// Variables
	float v_suma = 5;
	char *texto = "Ola k ace";
	int mi_entero = 123;
	char ch = '&';
	
	int n = 10;
	
	int a[n];
	for (int i=0; i<n; i++){
		a[i] = i;
		printf("a[%d] = %d\n", i, a[i]);
	}
	
	int b[n];
	for (int i=0; i<n; i++){
		b[i] = -i;
		printf("b[%d] = %d\n", i, b[i]);
	}
	
	int c[n];
	for (int i=0; i<n; i++){
		c[i] = a[i] + b[i];
		printf("c[%d] = %d\n", i, c[i]);
	}
	
	printf("%f, %s,\n %d, %c\n", v_suma, texto, mi_entero, ch);// %f -> flotante, %d -> entero, %s -> texto, %c -> caracter
	
	return suma(2.53,3.1); 
}

