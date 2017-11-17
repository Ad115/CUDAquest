#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int *buscarPalabra( char *sequence, char *buscar )
{


	//Conseguir los tamaños
	int tamSeq=strlen(sequence);
	int tamBuscar=strlen(buscar);

	//Conseguir el numero de ventanas
	int ventanas=tamSeq-tamBuscar+1;

	//Vector que guardara la posicion de los encontrados
	int *results=malloc(sizeof(int)*ventanas);
	//memset(results, 1, sizeof(int) * ventanas );

	int i,j;
	//Recorremos las ventanas
	for(i=0; i<ventanas; i++)
	{
		results[i]=1;
		//Recorre la pequeña cadena
		for( j=0; j<tamBuscar; j++ )
		{
			//Verificar si son la misma letra
			if ( sequence[i+j] != buscar[j] )
			{
				results[i]=0;
				break;
			}
		}
	}


	return results;
}

void imprimir( int *res, int n )
{
	printf("Se encontro: ");

	int i;
	for(i=0; i<n; i++)
	{
		if( res[i] == 1 )
			printf("%d ", i);
	}
	printf("\n");
}

int main( int argc, char *argv[] )
{
	//String en donde buscaremos una pequeña cadena
	//char *sequence="ACGATACCGATAGA";
	char *sequence=argv[1];

	//Pequeña cadena que buscaremos en la secuencia
	//char *buscar="GATA";
	char *buscar=argv[2];

	//Ejecutamos la busqueda de forma secuencial
	int *result=buscarPalabra( sequence, buscar );

	imprimir( result, strlen(sequence) - strlen(buscar) + 1 );


	free(result);
	return 0;
}
