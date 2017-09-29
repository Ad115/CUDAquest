//[[[[[[[[[[[[[[[[[[[[[[ 1 ]]]]]]]]]]]]]]]]]]]]]]




/*
El programa en C más simple
*/

int main() { return 0; }



//[[[[[[[[[[[[[[[[[[[[[[ 1 ]]]]]]]]]]]]]]]]]]]]]]



/*
Hola mundo
Para imprimir con printf:
    %d : entero
    %f : flotante
    %c : caracter
    %s : cadena de texto
*/

#include <stdio.h>

float Area(float base, float altura){
    return base * altura;
}

int Suma(int a, int b) {
    int suma = a + b;
    return suma;
}
int main() {
    int contador = 1;
    printf("El valor del contador es: %d\n", contador);

    float peque = 1.2e-3;
    printf("El valor del peque es: %f\n", peque);

    char letra = 'D';
    printf("El valor de la letra es: %d\n", letra);

    char nombre[] = "Ola k ace";
    printf("La cadena dice: %s\n", nombre);


    printf("Las variables: %d %f %c %s\n",
            contador,
            peque,
            letra,
            nombre
        );

    int i;
    for(i=0; i<=10; i++) {
        printf("i = %d\n", i);
    }

    int is_raining = 1;
    if(is_raining) {
        printf("Me mojaré\n");
    }

    // Funciones
    float area = Area(10.0, 15.0);
    printf("El área del rectángulo con base 10 y altura 15 es: %f\n", area);

    printf("La suma de 12 y 15 es: %d\n", Suma(12, 15));

    return 10;
}


//[[[[[[[[[[[[[[[[[[[[[[ 1 ]]]]]]]]]]]]]]]]]]]]]]

/*
Tipos de datos
    int (contar cosas)
    float (cosas no enteras)
    char (int disfrazado)
    char [] (cadena de texto)
*/
#include <stdio.h>
int main() {
    // Integer numbers ******************
    int an_integer = 24;
    int negative_integer = -24;

    printf("The integers are %d and %d \n",
        an_integer,
        negative_integer
    );

    // Floating point numbers ******************
    float small_number = 1.23e-3;
    float big_number = 123e+3;

    printf("The floating point numbers are %f and %f \n",
        small_number,
        big_number
    );

    // Characters ******************
    char A_Character = '$';
    int The_Character_Zero = '0';

    printf("The characters are %c and %c\n"
           "In integer form, they are %d and %d\n",
        A_Character,
        The_Character_Zero,
        A_Character,
        The_Character_Zero
    );

    // Strings
    char *string = "Hola mundo";
    char anotherString[] = "Hola mundo v2.0";
    char yetAnotherString[] = {'H', 'o', 'l', 'a', '\0'};
    //char *IKnowIKnow = { 'm', 'u', 'n', 'd', 'o', '\0' };

    printf("The first and second: %s and %s\n"
           "The second : %s mundo\n",
        string,
        anotherString,
        yetAnotherString
    );

    return 0;
}



//[[[[[[[[[[[[[[[[[[[[[[ 1 ]]]]]]]]]]]]]]]]]]]]]]

/*
Estructuras de control: If, If else
& valores booleanos
*/

#include <stdio.h>
int main() {

    char its_raining_outside;
    its_raining_outside = getchar();

    if (its_raining_outside == 'y') {
        printf("I'll go out to sing! :D \n");
    } else {
        printf("I'll burn in Qro's sun :P\n");
    }

    return 0;
}


//[[[[[[[[[[[[[[[[[[[[[[ 1 ]]]]]]]]]]]]]]]]]]]]]]


//[[[[[[[[[[[[[[[[[[[[[[ 1 ]]]]]]]]]]]]]]]]]]]]]]
