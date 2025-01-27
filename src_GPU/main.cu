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

    if(opts.tauSim == 1)
    {
        if(opts.verbose) printOptions(&opts);
    }

    if(opts.verbose) printOptions(&opts);

    if(opts.nD == 3)
    {
        if(opts.SteadyStateFlag == 1)
        {
            SteadyStateSim3D(&opts);
        }
    }
    else if(opts.nD == 2)
    {
        if(opts.SteadyStateFlag == 1)
        {
            SteadyStateSim2D(&opts);
        }
    }
    
    return 0;
}