#include "stdlib.h"
#include "stdio.h"
#include "fluidMethod.h"

int main(int argc, char **argv)
{
#ifdef USE_SSPRK
printf("4\n"); // four cells
#else
#ifdef USE_RK3
printf("6\n"); // six cells
#else
printf("3\n"); // three cells
#endif
#endif
return 0;
}
