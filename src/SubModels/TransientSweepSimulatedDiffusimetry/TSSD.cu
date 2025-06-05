/*

TSSD Submodel Main File:


This file will run most of the TSSD subroutines for
estimating the diffusion coefficient in the active material
by matching data from the neutron experiment.


Andre Adam.

Last Updated:

05/13/2025
*/

#include <TSSD.cuh>

int main(int argc, char **argv)
{
    // Declare structs
    options opts;
    TSSDopts oTSSD;
    meshInfo mesh;
    Migration mig;

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

    // read mig if necessary
    if (oTSSD.useMig)
    {
        bool error = readInputMig(&mig);
        if(error)
        {
            printf("Some error occured while trying to read the input file for Migration model.\n");
            printf("Exiting Now!\n");
            return 1;
        }
    }

    // print options

    if (opts.verbose)
        printTSSD(&opts, &oTSSD, &mig);


    // Load image to simulate

    char *simData;

    readImg2D(&opts, &mesh, simData);

    // set mesh parameters

    mesh.dx = oTSSD.pixelRes / opts.MeshIncreaseX;
    mesh.dy = oTSSD.pixelRes / opts.MeshIncreaseY;

    // Automatically find dt

    double maxDC = 0;

    for (int i = 0; i < opts.numDC; i++)
    {
        if (i == 0 && opts.DC[i] != 0)
            maxDC = opts.DC[i];
        else if (opts.DC[i] != 0 && opts.DC[i] > maxDC)
            maxDC = opts.DC[i];
    }

    mesh.dt = 20 * mesh.dx * mesh.dx / maxDC;
    if (oTSSD.useMig)
    {
        if (fabs(mesh.dx * maxDC * opts.charge * FARADAY / (GAS_C * mig.T) * mig.dE_dL[1]) > maxDC)
        {
            double temp = fabs(mesh.dx * maxDC * opts.charge * FARADAY / (GAS_C * mig.T) * mig.dE_dL[1]);
            mesh.dt = 20 * mesh.dx * mesh.dx / temp;
        }
    }

    if (opts.verbose)
    {
        printf("Pixel Res = %1.3e\n", mesh.dx);
        printf("Mesh DT = %1.3e\n", mesh.dt);
    }

    // if multiple methods are selected, just return an error

    if(oTSSD.useGITT + oTSSD.useLinear + oTSSD.useAnom > 1)
    {
        printf("Multiple models selected, returning...\n");
        return 1;
    }

    // Create arrays for BC's and DC's

    double *DC = (double *)malloc(sizeof(double) * mesh.nElements);
    int *BC = (int *)malloc(sizeof(int) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));
    double *BC_Value = (double *)malloc(sizeof(double) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));

    double *GITT_D;
    double *GITT_SOC;
    int nData = 100;    // just a hardcoded default, assuming I don't have more than 100 GITT points

    if (oTSSD.useGITT)
    {
        // create space for arrays
        GITT_D = (double *)malloc(sizeof(double) * nData);
        GITT_SOC = (double *)malloc(sizeof(double) * nData);
        //  set memory
        memset(GITT_D, 0, sizeof(double) * nData);
        memset(GITT_SOC, 0, sizeof(double) * nData);
        // read GITT data
        GITT_Interval(&oTSSD, GITT_SOC, GITT_D, &nData);
    }

    // initialize arrays

    memset(DC, 0, sizeof(double) * mesh.nElements);
    memset(BC, 0, sizeof(int) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));
    memset(BC_Value, 0, sizeof(double) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));

    // start an array for SOC

    double SOC = 0;
    int GITT_idx = 0;
    double POI_DC = 0;

    if (oTSSD.useGITT == 0)
        SetDC2D(&opts, &mesh, DC, simData);
    else
    {
        POI_DC = GITT_D[GITT_idx];
        SetDC_GITT(&opts, &oTSSD, &mesh, DC, simData, POI_DC);
    }

    // BC Conditions for TSSD Model

    activeSA_2D(&opts, &mesh, DC);

    SetBC_TSSD2D(&opts, &oTSSD, &mesh, BC, BC_Value);

    // Flood-Fill Bottom Start
    FloodFill2D_Bot(&mesh, BC, DC);

    // Load data to match

    /*
        Not there yet
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
    memset(C0, 0.0, sizeof(double) * mesh.nElements);

    for(int i = 0; i < mesh.nElements; i++)
    {
        if (DC[i] == 0)
            continue;
        Concentration[i] = oTSSD.C0;    // mol/m^3
        C0[i] = oTSSD.C0;               // mol/m^3
    }

    if(oTSSD.useAnom)
    {
        // populate DC array
        setDC_AnomDiff(&oTSSD, &mesh, DC, Concentration, simData);
    }

    
    // if using linear model, update

    if(oTSSD.useLinear)
    {
        SetDC_Linear(&opts, &oTSSD, &mesh, DC, simData, Concentration);
    }

    // Declare needed arrays

    double *d_Coeff = NULL;
    double *d_RHS = NULL;
    double *d_Conc = NULL;
    double *d_ConcTemp = NULL;

    // Now we confirm that there is a match in GPUs available and user expectations

    if (opts.useGPU) 
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

    // Migration contribution to discretization
    if(oTSSD.useMig)
        Disc_Mig2D(CoeffMatrix, DC, RHS, C0, &opts, &mesh, &mig);

    mesh.currentTime = 0;

    int step = 0;

    double timeToCheck = oTSSD.stepSize;

    // save C(y,t)

    saveCyt(&mesh, Concentration, step);

    while (mesh.currentTime <= oTSSD.totalTime)
    {
        // if using GITT data, check for updates to DC
        if(oTSSD.useGITT)
        {
            SOC = mesh.currentTime / oTSSD.totalTime * 100;
            if (SOC >= GITT_SOC[GITT_idx + 1] && GITT_SOC[GITT_idx + 1] != 0 && GITT_D[GITT_idx + 1] != 0)
            {
                // update DC
                GITT_idx++;
                POI_DC = GITT_D[GITT_idx];
                SetDC_GITT(&opts, &oTSSD, &mesh, DC, simData, POI_DC);
                // discretize system again
                DiscTrans2D(&opts, &mesh, BC, BC_Value, DC, CoeffMatrix, RHS, C0);
                printf("Updated DC: %1.3e, Time = %1.3e, SOC = %1.3e\n", POI_DC, mesh.currentTime, SOC);
            }
            else if(mesh.currentTime != 0)
            {
                // no DC update, only update RHS
                RHS_Update2D(&mesh, BC, BC_Value, CoeffMatrix, RHS, C0);
            }
        } 
        else if(oTSSD.useLinear)
        {
            // update diffusion coefficients
            SetDC_Linear(&opts, &oTSSD, &mesh, DC, simData, Concentration);
            // discretize system again
            DiscTrans2D(&opts, &mesh, BC, BC_Value, DC, CoeffMatrix, RHS, C0);
        }
        else if(oTSSD.useAnom)
        {
            // update DC and discretize again
            setDC_AnomDiff(&oTSSD, &mesh, DC, Concentration, simData);

            DiscTrans2D(&opts, &mesh, BC, BC_Value, DC, CoeffMatrix, RHS, C0);
        }
        else
        {
            // not using GITT data
            if (mesh.currentTime != 0)
            {
                // coefficient matrix is still good, just update the RHS
                // RHS_Update2D(&mesh, BC, BC_Value, CoeffMatrix, RHS, C0);
                DiscTrans2D(&opts, &mesh, BC, BC_Value, DC, CoeffMatrix, RHS, C0);
                Disc_Mig2D(CoeffMatrix, DC, RHS, C0, &opts, &mesh, &mig);
            }
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
            if (mesh.currentTime == 0)
            {
                JI2D_SOR(CoeffMatrix, RHS, Concentration, d_Coeff,
                         d_RHS, d_Conc, d_ConcTemp, &opts, &mesh);
            }
            else
            {
                JI2D_TransientUpdate(RHS, Concentration, d_Coeff,
                                     d_RHS, d_Conc, d_ConcTemp, &opts, &mesh);
            }
        }

        // Update time
        mesh.currentTime += mesh.dt;

        // Copy new concentration into C0
        memcpy(C0, Concentration, sizeof(double) * mesh.nElements);

        // save data if necessary
        if (mesh.currentTime > timeToCheck)
        {
            timeToCheck += oTSSD.stepSize;
            step++;

            saveCyt(&mesh, Concentration, step);
            if (opts.verbose)
                printf("Time = %1.3e\n", mesh.currentTime);
            
            // check NaN's
            for(int i = 0; i < mesh.nElements; i++)
            {
                if(Concentration[i] != Concentration[i])
                {
                    printf("Found NaN at %d, time %1.3e\n", i, mesh.currentTime);
                    return 1;
                }
            }
        }
    }

    printCandF(&opts, &oTSSD, &mesh, DC, Concentration);

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

    // Manage GPU Memory (if applicable)

    if (opts.useGPU)
    {
        unInitGPU_SOR(&d_Coeff, &d_RHS, &d_Conc, &d_ConcTemp);
    }

    // Memory management

    free(CoeffMatrix);
    free(Concentration);
    free(C0);
    free(RHS);

    free(BC);
    free(BC_Value);
    free(DC);

    return 0;
}