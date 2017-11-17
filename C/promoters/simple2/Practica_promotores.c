#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BUFF_INI 128

int windowNumber( int sequenceLen, int windowSize );
float analyzeWindow(char *sequence, int winSize);
void sequenceWindows( char *sequence, int winSize, int winNum, FILE *salida);
float energyA(char nucleotide);
float energyC(char nucleotide);
float energyG(char nucleotide);
float energyT(char nucleotide);
float energias(char nucl1, char nucl2);
char *redimensionaBuffer(char *buffer, int size);
void help(void);

int main( int argc, char *argv[] )
{
	FILE *sequenceFile = NULL;
	FILE *energyOut    = NULL;
	char *entrada = NULL;
  char *salida  = NULL;
	int ventana = 8;
	int i;
	int windowNum;

	char *sequence = NULL;
	for(i=1; i < argc; i++){

		if ((strcmp(argv[i],"-h") == 0) || (strcmp(argv[i],"--help") == 0)){
			help();
		} else if ( strcmp(argv[i],"-i") == 0 ) {
			entrada = argv[++i];
		} else if ( strcmp(argv[i],"-o") == 0 ) {
			salida = argv[++i];
		} else if ( strcmp(argv[i],"-k") == 0 ) {
			ventana = atoi(argv[++i]);
		} else{
			printf("Opcion no valida %s\n", argv[i]);
			exit(0);
		}

	}

	sequenceFile  = fopen(entrada,"r");
	energyOut     = fopen(salida,"w");
	if(sequenceFile == NULL || energyOut == NULL){
		printf("Error:\n\tNo se puede accesar al archivo entrada/salida\n");
		exit(0);
	}

	sequence = (char *)malloc(sizeof(char) * BUFF_INI);

	if( sequence == NULL ){
		printf("Error:\n\tNo existe suficiente memoria\n");
		exit(0);
	}

	i = 0;

	while( fread(sequence+i,1,1,sequenceFile) == 1 ) {
		if(sequence[i] == '\n'){
			sequence[i] = '\0';
			if(sequence[0] == '>'){
				fprintf(energyOut, "%s\n", sequence);
			} else {
				windowNum = windowNumber( i, ventana );
				sequenceWindows( sequence, ventana, windowNum, energyOut);
			}
			i = 0;
			continue;
		}

		sequence = redimensionaBuffer(sequence,i);
		i++;
	}

	free(sequence);
	fclose(sequenceFile);
	fclose(energyOut);


	return 0;
}



int windowNumber( int sequenceLen, int windowSize )
{
	return ( sequenceLen - windowSize + 1);
}


float analyzeWindow(char *sequence, int winSize)
{
	float energy=0;
	int i;
	for(i=0; i<winSize-1; i++)
	{
		energy+=energias( sequence[i], sequence[i+1] );
	}
	return energy;
}

void sequenceWindows( char *sequence, int winSize, int winNum, FILE *salida)
{
	int i;
	float window;

	for( i=0; i<winNum; i++ )
	{
		window=analyzeWindow( ( sequence + i ), winSize );
		fprintf(salida,"%f\n",window);
	}
}

float energyA(char nucleotide)
{
	if( nucleotide == 'C' )
	{
		return -7.5712;
	}

	else if( nucleotide == 'G' )
	{
		return -7.023;
	}
	else if( nucleotide == 'T' )
	{
		return -6.4452;
	}
	else
	{
		return -7.0786;
	}
}

float energyC(char nucleotide)
{
	if( nucleotide == 'A' )
	{
		return -7.6601;
	}

	else if( nucleotide == 'G' )
	{
		return -9.5936;
	}
	else if( nucleotide == 'T' )
	{
		return -7.023;
	}
	else
	{
		return -7.2637;
	}
}

float energyG(char nucleotide)
{
	if( nucleotide == 'A' )
	{
		return -7.3786;
	}

	else if( nucleotide == 'C' )
	{
		return -8.8972;
	}
	else if( nucleotide == 'T' )
	{
		return -7.5712;
	}
	else
	{
		return -7.2637;
	}
}

float energyT(char nucleotide)
{
	if( nucleotide == 'A' )
	{
		return -6.4119;
	}

	else if( nucleotide == 'C' )
	{
		return -7.3786;
	}
	else if( nucleotide == 'G' )
	{
		return -7.6601;
	}
	else
	{
		return -7.0786;
	}
}

float energias(char nucl1, char nucl2)
{
	if( nucl1 == 'A' )
	{
		return energyA(nucl2);
	}

	else if( nucl1 == 'C' )
	{
		return energyC(nucl2);

	}
	else if( nucl1 == 'G' )
	{
		return energyG(nucl2);

	}
	else
	{
		return energyT(nucl2);

	}
}

char *redimensionaBuffer(char *buffer, int size){
	if(size == (BUFF_INI - 1)){
		buffer = (char *)realloc(buffer,sizeof(char) * (BUFF_INI + BUFF_INI/4));
		if( buffer == NULL ){
			printf("Error:\n\tNo existe suficiente memoria para realocar\n");
			exit(0);
		}

	}

	return buffer;
}

void help(void){
  printf(
    "\n"
    "USO\n"
    "   ./promotores [-k tamano_ventana] -i archivo.fa -o salida.tsv\n\n"
  );
  exit(0);
}
