#include <ASSC.cuh>

int main(int argc, char **argv)
{
    // declare structs
    options opts;
    ASSCopts oASSC;

    // read input data
    
    char inputName[100];
    sprintf(inputName, "inputASSC.txt");

    // Check if file exists

    bool fileExist = false;

    // Check if file exists

    if (FILE *TEST = fopen(inputName, "r"))
    {
        fclose(TEST);
        fileExist = true;
    }

    if (!fileExist)
    {
        printf("Input file not found, returning...\n");
        return 1;
    }

    // read input (and print?)

    readInputGeneral(inputName, &opts);
    readInputASSC(inputName, &oASSC);

    if (opts.verbose)
        printInputASSC(&opts, &oASSC);

    test_funct();

    return 0;
}