#include <stdio.h>

__global___ kernel(void)    {}

int main()
{
    kernel<<<1,1>>>();
    printf("Hello world!\n")
}
