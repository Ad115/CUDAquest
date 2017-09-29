#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b){
	printf("a: %d, b:%d\n", *a, *b);
	int tmp = *b;
	*b = *a;
	*a = tmp;
	printf("a: %d, b:%d\n", *a, *b);
}

int main() {
	int n=10;
	int *puntero = &n;
	printf("El número es : %d\n", n);
	printf("El valor de puntero es: %p\n", puntero);
	printf("El valor al que apunta puntero es: %d\n", *puntero);
	
	int arreglo[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
	int *puntero2 = arreglo;
	printf("El valor de puntero2 es: %p\n", puntero2);
	printf("El valor al que apunta puntero es: %d\n", *puntero2);
	
	printf("El 3er elemento de arreglo es: %d\n", arreglo[2]);
	printf("El 3er elemento de arreglo es: %d\n", *(arreglo+2));
	
	int a = 1; int b = 3;
	swap(&a, &b);
	printf("Afuera... a: %d, b:%d\n", a, b);
	
	int *arreglo2 = malloc(1000 * sizeof(int)); 
	
	for (int i=0; i<1000; i = i+1) {
		arreglo2[i] = i*i*i;
	}
	for (int i=0; i<1000; i++) {
		printf("Elemento número %d : %d\n", i, arreglo2[i]);
	}
	free(arreglo2);
}