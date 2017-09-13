#include "stdlib.h"
#include "stdio.h"
#include "fluidMethod.h"

int main(int argc, char **argv)
{
#ifdef USE_SSPRK
printf("1\n");
#else
printf("0\n");
#endif
return 0;
}
