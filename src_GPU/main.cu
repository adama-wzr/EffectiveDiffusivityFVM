#include "helper.cuh"

int main(int argc, char **argv)
{
    // Important call for efficiency on Linux
	fflush(stdout);

	//	Declare data structure
	options opts;

	char inputFilename[30];

	sprintf(inputFilename, "input.txt");

    readInputGeneral(inputFilename, &opts);

    // Do some checks to make sure the input was ok

    

    if(opts.verbose) printOptions(&opts);

    printf("Num D's expected = %d\n", opts.numDC);

    for(int i = 0; i < opts.numDC; i++)
    {
        printf("D%d = %1.3e\n",i+1, opts.DC[i]);
        printf("D_TH%d = %d\n", i+1, opts.DC_TH[i]);
    }
    
    return 0;
}