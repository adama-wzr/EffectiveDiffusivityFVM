#include "helper.cuh"

int main(int argc, int *argv[])
{
    // set env variables
    fflush(stdout);
    // Main file will call different models
    test();
    
    return 0;
}