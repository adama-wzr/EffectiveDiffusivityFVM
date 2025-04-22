/*

TSSD Submodel Main File:


This file will run most of the TSSD subroutines for
estimating the diffusion coefficient in the active material
by matching data from the neutron experiment.


Andre Adam.

Last Updated:

04/21/2025
*/


#include <TSSD.cuh>


int main(int argc, char **argv)
{
    // Declare structs
    options opts;
    TSSDopts oTSSD;
    
    // TSSD Input Name

    char inputFilename[50];

	sprintf(inputFilename, "inputTSSD.txt");

    // Check if file exists

    bool fileExist = false;

    // Check if file exists

    if (FILE *TEST = fopen(inputFilename, "r"))
    {
        fclose(TEST);
        fileExist = true;
    }

    if (!fileExist)
    {
        printf("Input file not found, returning...\n");
        return 1;
    }

    // read input

    readInputGeneral(inputFilename, &opts);

    // read input TSSD
    readInputTSSD(inputFilename, &oTSSD);

    // print options

    if(opts.verbose)
        printTSSD(&opts, &oTSSD);

    // Pseudo-Code

    // Load image to simulate

    // Remove non-participating media

    // Load data to match

    // Simulate 5 minutes at different Diffusion coefficients

    // Pick simulations that match the concentration profile by some metric

    // Interpolate the actual diffusion coefficient

    // Simulate the new concentration distribution based on theory of anomalous diffusion
    // and using the previous value statically.

    // Assess these two results, interpolate, find a new average coefficient.
    /*
        If new coefficient is similar to anomalous diffusion theory, then don't use it.

        If new coefficient is better, then use it to simulate concentration map again.

        NOTES:
            - what metric to use ?
            - What is good enough ?
    */

    // repeat these steps for a full charge cycle, full discharge cycle.

    /*
        Data to be saved:
            - save average Lithium concentration in y-direction
            - save Li concentration maps every 5 minutes.
            - Let's use a small domain for this simulation.
    */



    return 0;
}