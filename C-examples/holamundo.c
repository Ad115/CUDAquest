/*
 Esto está comentado
 
 y esto tmb xD
*/

#include <stdio.h>

/*
 * Función suma
 */
float suma(float a, float b){
	
	return a+b;
}

int main() {
	// Variables de diferetes tipos
	float v_suma = 5;
	char *texto = "Ola k ace";
	int mi_entero = 123;
	char ch = '&';
	int n = 10;
	
    // Inicializa e imprime el arreglo a (a[i] = i)
	int a[n];
	for (int i=0; i<n; i++){
		a[i] = i;
		printf("a[%d] = %d\n", i, a[i]);
	}
	
	// Inicializa e imprime el arreglo b (b[i] = -i)
	int b[n];
	for (int i=0; i<n; i++){
		b[i] = -i;
		printf("b[%d] = %d\n", i, b[i]);
	}
	
	// Inicializa e imprime el arreglo c (c[i] = a[i] + b[i] = 0)
	int c[n];
	for (int i=0; i<n; i++){
		c[i] = a[i] + b[i];
		printf("c[%d] = %d\n", i, c[i]);
	}
	
	// Imprime distintos tipos de variables
	printf("%f, %s,\n %d, %c\n", v_suma, texto, mi_entero, ch);// %f -> flotante, %d -> entero, %s -> texto, %c -> caracter
	
    // Checa el valor de retorno en la terminal ejecutando `echo $?`
	return suma(2.53,3.1); 
}

