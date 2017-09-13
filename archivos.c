/* 
Primer programa de entrada / salida
-----------------------------------

Abre un archivo para lectura y lee la primera línea.

Este programa introduce 4 conceptos principales:

	1. El tipo de dato FILE.
	2. La función fopen.
	3. La función fscanf para leer desde un archivo.
	4. La función fclose.
*/


#include <stdio.h>

int main() {
	
	// Abre el archivo para lectura
	FILE *file = fopen("nombre-edad.tsv", "r"); // Los modos más comunes son:
												// - Lectura: 'r'
												// - Escritura: 'w'
												// - Concatenación: 'c'
			
		char nombre[100]; // Un nombre leído del archivo
		int edad; // La edad leída del archivo
		
		// Lee el primer par (nombre, edad)
		fscanf( file,  "%s %d",  nombre, &edad );
			
		// Imprímelos a la pantalla
		printf("Nombre: %s, Edad: %d\n", nombre, edad);
		
	
	// Cierra el archivo
	fclose(file);
	
	return 0;
}

/* 
Segundo programa de entrada / salida
------------------------------------

Abre un archivo para lectura y lee todas las líneas.

El programa introduce 2 conceptos principales:

	1. El marcador de fin de archivo EOF
	2. la función feof
*/


#include <stdio.h>

int main() {
	
	// Abre el archivo para lectura
	FILE *file = fopen("nombre-edad.tsv", "r");
			
		char nombre[100]; // Un nombre leído del archivo
		int edad; // La edad leída del archivo
		
		// Escanea todas las líneas hasta el EOF
		while( ! feof(file) ) { 
			// Obtén un nuevo par nombre, edad
			fscanf( file,  "%s %d",  nombre, &edad );
			
			// Imprímelos a la pantalla
			printf("Nombre: %s, Edad: %d\n", nombre, edad);
		}
	
	// Cierra el archivo
	fclose(file);
	
	return 0;
}


/* 
Tercer programa de entrada / salida
-----------------------------------

Escanea todas las edades en el archivo, ordénalas e imprímelas ordenadas

El siguiente programa repasa los conceptos anteriores.
*/


#include <stdio.h>

void bubbleSort(int *arreglo, int n) {
/*
 * Ordena el arreglo `arreglo`  por bubble sort,
 * asumiendo que tiene `n` entradas.
 */
	int tmp; // Variable temporal
	
	for(int i=0; i < n-1; i++) {
		// Checa si la entrada i e i+1 están ordenadas
		if (arreglo[i] > arreglo[i+1]) {
			// No están ordenadas, entonces ordénalas
			tmp = arreglo[i];
			arreglo[i] = arreglo[i+1];
			arreglo[i+1] = tmp;
			// Vuelve a checar desde el inicio
			i = 0 - 1;
		}
		
	}
}


int main() {
	
	// Abre el archivo para lectura
	FILE *file = fopen("nombre-edad.tsv", "r");
			
		char nombre[100]; // Un nombre leído del archivo
		int edades[50]; // La edad leída del archivo
		int n=0; // Contador para las edades
		
		// Escanea todas las líneas hasta el EOF 
		// o hasta leer las edades permitidas
		while( !feof(file) && n<50 ) { 
			
			// Obtén un nuevo par nombre, edad
			fscanf( file,  "%s %d",  nombre, &(edades[n]) );
			
			// Imprímelos a la pantalla
			printf("Nombre: %s, Edad: %d\n", nombre, edades[n]);
			
			// Incrementa el contador
			n++;
		}
	
	// Ya terminaste con el archivo, ciérralo
	fclose(file);
	
	// Ordena el arreglo de las edades
	bubbleSort(edades, n);
	
	// Imprime el arreglo de las edades
	for(int i=0; i<n; i++) {
		// Imprime la edad
		printf("%d. %d\n", i, edades[i]);
	}
	
	return 0;
}


/* 
Cuarto programa de entrada / salida
-----------------------------------

Abre un archivo para escritura y escribe una línea.

El siguiente programa introduce 2 conceptos principales:

	1. El segundo parámetro de la función fopen
	2. La función fprintf
*/


#include <stdio.h>

int main() {
	
	// Abre el archivo para escritura
	FILE *file = fopen("salida.tsv", "w");
			
		char *nombre = "Light Yagami"; // Un nombre 
		int edad = 22; // La edad 
		
		// Escribe en el archivo el par nombre, edad
		fprintf( file,  "%s\t%d\n",  nombre, edad );
			
		// Imprímelos a la pantalla
		printf("Se ha escrito '%s\t%d' al archivo\n", nombre, edad);
	
	// Cierra el archivo
	fclose(file);
	
	return 0;
}



/* 
Quinto programa de entrada / salida
-----------------------------------

Abre un archivo para lectura, lee las edades allí contenidas,
ordénalas mediante bubble sort y escríbelas en otro archivo.

El siguiente programa integra los conceptos anteriores.
*/


#include <stdio.h>

void bubbleSort(int *arreglo, int n) {
/*
 * Ordena el arreglo `arreglo`  por bubble sort,
 * asumiendo que tiene `n` entradas.
 */
	int tmp; // Variable temporal
	
	for(int i=0; i < n-1; i++) {
		// Checa si la entrada i e i+1 están ordenadas
		if (arreglo[i] > arreglo[i+1]) {
			// No están ordenadas, entonces ordénalas
			tmp = arreglo[i];
			arreglo[i] = arreglo[i+1];
			arreglo[i+1] = tmp;
			// Vuelve a checar desde el inicio
			i = 0 - 1;
		}
		
	}
}


int main() {
	
	// Abre el archivo para lectura
	FILE *file = fopen("nombre-edad.tsv", "r");
			
		char nombre[100]; // Un nombre leído del archivo
		int edades[50]; // La edad leída del archivo
		int n=0; // Contador para las edades
		
		// Escanea todas las líneas hasta el EOF 
		// o hasta leer las edades permitidas
		while( !feof(file) && n<50 ) { 
			
			// Obtén un nuevo par nombre, edad
			fscanf( file,  "%s %d",  nombre, &(edades[n]) );
			
			// Imprímelos a la pantalla
			printf("Nombre: %s, Edad: %d\n", nombre, edades[n]);
			
			// Incrementa el contador
			n++;
		}
	
	// Ya terminaste con el archivo, ciérralo
	fclose(file);
	
	// Ordena el arreglo de las edades
	bubbleSort(edades, n);
	
	// Abre el archivo de salida (reciclamos el puntero)
	file = fopen("salida.txt", "w");
	
		// Escribe las edades ordenadas en el archivo
		for(int i=0; i<n; i++) {
			// Imprime la edad
			fprintf( file, "%d\n", edades[i]);
		}
		
	// Terminaste, cierra el archivo
	fclose(file);
	
	return 0;
}