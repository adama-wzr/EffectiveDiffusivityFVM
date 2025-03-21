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
        if(opts.verbose) printOpts_Tau(&opts);
        if(opts.nD == 3)
        {
            Tau3D_Sim(&opts);
        } else if(opts.nD == 2)
        {
            Tau2D_Sim(&opts);
        }
    }
    else if(opts.nD == 3 && opts.SteadyStateFlag == 1)
    {
        if(opts.verbose) printOptions(&opts);
        if(opts.SteadyStateFlag == 1)
        {
            SteadyStateSim3D(&opts);
        }
    }
    else if(opts.nD == 2 && opts.SteadyStateFlag == 1)
    {
        if(opts.verbose) printOptions(&opts);
        if(opts.SteadyStateFlag == 1)
        {
            SteadyStateSim2D(&opts);
        }
    }

    if (opts.TF_Flag)
    {
        if (opts.verbose)
            printOptions(&opts);
        if (opts.nD == 2)
            TransientFluxSim2D(&opts);
    }

    return 0;
}