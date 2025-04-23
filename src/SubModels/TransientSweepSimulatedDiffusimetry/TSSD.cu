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
    meshInfo mesh;
    
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

    char *simData;

    readImg2D(&opts, &mesh, simData);

    // set mesh parameters

    mesh.dx = oTSSD.pixelRes / mesh.numCellsX;
    mesh.dy = oTSSD.pixelRes / mesh.numCellsY;

    // Automatically find dt

    double maxDC = 0;

    for(int i = 0; i <  opts.numDC; i++)
    {
        if(i == 0 && opts.DC[i] != 0)
            maxDC = opts.DC[i];
        else if( opts.DC[i] != 0 && opts.DC[i] > maxDC)
            maxDC = opts.DC[i];
    }

    mesh.dt = 10 * mesh.dx*mesh.dx/maxDC;

    // Create arrays for BC's and DC's

    double *DC = (double *)malloc(sizeof(double) * mesh.nElements);
    int *BC = (int *)malloc(sizeof(int) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));
    double *BC_Value = (double *)malloc(sizeof(double) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));

    // initialize arrays

    memset(DC, 0, sizeof(double) * mesh.nElements);
    memset(BC, 0, sizeof(int) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));
    memset(BC_Value, 0, sizeof(double) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));

    SetDC2D(&opts, &mesh, DC, simData);

    // BC Conditions for TSSD Model

    activeSA_2D(&opts, &mesh, DC);

    SetBC_TSSD2D(&opts, &oTSSD, &mesh, BC, BC_Value);

    // Flood-Fill Bottom Start
    FloodFill2D_Bot(&mesh, BC, DC);

    // Load data to match

    /*
        Not there yet
    */

    // Simulate 5 minutes at different Diffusion coefficients

    /*
        Let's simulate using the DC of the first GITT step.
    */

    // Allocate arrays for holding discretized equations

    double *CoeffMatrix = (double *)malloc(mesh.nElements * 5 * sizeof(double));
    double *RHS = (double *)malloc(mesh.nElements * sizeof(double));
    double *Concentration = (double *)malloc(mesh.nElements * sizeof(double));

    double *C0 = (double *)malloc(sizeof(double) * mesh.nElements);

    // initialize the memory

    memset(CoeffMatrix, 0.0, mesh.nElements * sizeof(double) * 5);
    memset(RHS, 0.0, mesh.nElements * sizeof(double));
    memset(Concentration, 0.0, mesh.nElements * sizeof(double));
    memset(C0, 0.0, sizeof(double) * mesh.nElements);     // unless we pass a field-function, C0 = 0 is fine

    // Declare needed arrays

    double *d_Coeff = NULL;
    double *d_RHS = NULL;
    double *d_Conc = NULL;
    double *d_ConcTemp = NULL;

    // Now we confirm that there is a match in GPUs available and user expectations

    if(opts.useGPU)
    {
        int nDevices;
        cudaGetDeviceCount(&nDevices);

        if (nDevices < 1)
        {
            printf("No CUDA-capable GPU Detected! Exiting...\n");
            return 1;
        }
        else if (nDevices < opts.nGPU)
        {
            printf("User requested %d GPUs, but only %d were detected.\n", opts.nGPU, nDevices);
            printf("Proceeding with %d GPUs\n", nDevices);
            opts.nGPU = nDevices;
        }

        // Initialize the GPU arrays

        initGPU_2DSOR(&d_Coeff, &d_RHS, &d_Conc, &d_ConcTemp, &mesh);
    }

    // New discretization needed
    DiscTrans2D(&opts, &mesh, BC, BC_Value, DC, CoeffMatrix, RHS, C0);

    mesh.currentTime = 0;

    int step = 0;

    double timeToCheck = oTSSD.stepSize;

    // save C(y,t)

    saveCyt(&mesh, Concentration, step);

    while(mesh.currentTime <= oTSSD.totalTime)
    {
        if (mesh.currentTime != 0)
        {
            // coefficient matrix is still good, just update the RHS
            RHS_Update2D(&mesh, BC, BC_Value, CoeffMatrix, RHS, C0);
        }

        if (opts.useGPU == 0)
        {
            // CPU Solve
            omp_set_num_threads(opts.nThreads);

            GS2D_OMP(CoeffMatrix, RHS, Concentration, &opts, &mesh);
        }
        else
        {
            // GPU Solve

            JI2D_TransientUpdate(RHS, Concentration, d_Coeff,
                                    d_RHS, d_Conc, d_ConcTemp, &opts, &mesh);
        }


        // Update time
        mesh.currentTime+=mesh.dt;
        
        // save data if necessary
        if(mesh.currentTime > timeToCheck)
        {
            timeToCheck += oTSSD.stepSize;
            step++;

            saveCyt(&mesh, Concentration, step);
            if (opts.verbose)
                printf("Time = %1.3e\n", mesh.currentTime);
        }
    }

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