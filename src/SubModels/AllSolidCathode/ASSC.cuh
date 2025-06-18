#ifndef _ASSC
#define _ASSC

#include <math.h>
#include <fstream>
#include <iostream>
#include <string>

#include <helper.cuh>
#include <Migration.cuh>
#include <datastructures.cpp>
#include <constants.cpp>

/*

    Read input, assign defaults, print if necessary

*/

void printInputASSC(options *opts, ASSCopts *oASSC, meshInfo *mesh)
{
    /*
        printInputASSC Function:
        Inputs:
            - pointer to options struct
            - pointer to ASSC struct
        Outputs:
            - None.
        
        Function will print user entered options and
        defaults that have been stored in the data structures
        relevant to this simulation type.
    */

    // Print Model Options

    printf("--------------------------------------------\n\n");
    printf("            ASSC Model Inputs:              \n\n");
    printf("--------------------------------------------\n\n");

    printf("Input image name: %s\n", opts->inputFilename);

    printf("POI: %1.0d, Grayscale Threshhold = %3.0d\n", oASSC->POI, oASSC->POI_TH);

    printf("POI DC: %1.3e m^2/s\n", oASSC->POI_DC);

    printf("Initial concentration: %1.3e [mol/m^3]\n", oASSC->C0);

    printf("Pixel Resolution: %1.3e\n", oASSC->pixelRes);

    printf("Current Density: %1.3f A/m^2\n", oASSC->currentDensity);

    if (oASSC->C_or_D)
        printf("Simulating Charge Step\n");
    else
        printf("Simulating Discharge Step\n");

    printf("Start: %5.1lf (s), step: %4.1lf (s), final: %5.1lf (s)\n",
        oASSC->startTime, oASSC->stepTime, oASSC->totalTime);

    printf("--------------------------------------------\n\n");
    printf("            Simulation Options:             \n\n");
    printf("--------------------------------------------\n\n");

    // mesh amplificaiton

    printf("Mesh Refine X = %d\n", opts->MeshIncreaseX);
    printf("Mesh Refine Y = %d\n", opts->MeshIncreaseY);
    if (opts->nD == 3)
    {
        printf("Mesh Refine Z = %d\n", opts->MeshIncreaseZ);
    }

    // number of cells

    printf("nElements: %ld\n", mesh->nElements);
    printf("AM VF %1.3e\n", oASSC->AM_VF);
    printf("AM SA: %1.3e m^2\n", mesh->SA);
    printf("AM SSA: %1.3e m^-1\n", mesh->SSA);

    // Acceleration

    printf("Computation Mode:\n");

    if (opts->useGPU == 1)
    {
        printf("Using %d GPU(s)\n", opts->nGPU);
    }
    else
    {
        printf("Number of Threads = %d\n", opts->nThreads);
    }

    // convergence

    printf("Max. Iterations: %ld\n", opts->MAX_ITER);
    printf("Convergence: %1.3e\n", opts->ConvergeCriteria);
    printf("Time Step: %1.3e [s]\n", mesh->dt);


    // BC
    if(oASSC->PB)
        printf("BC = periodic\n");
    else
        printf("BC = no flux (sides)\n");


    return;
}

void readInputASSC(char *FileName, ASSCopts *oASSC)
{

    /*
        readInputASSC Function:
        Inputs:
            - FileName: pointer to where the input file name is stored.
            - ASSCopts: data structure options for ASSC simulation
        Outputs: None

        Function reads the input file for ASSC.
    */

    // initiate necessary variables for input reading
    std::string myText;

    char tempC[1000];
    double tempD;
    char tempFilenames[1000];
    std::ifstream InputFile(FileName);

    char tempDC[20];
    char tempDC_TH[20];
    int DC_read = 0;
    int DC_TH_read = 0;

    // Default values set here
    oASSC->POI = 1;
    oASSC->POI_TH = 150;
    oASSC->POI_DC = 1e-10;  // m^2/s
    oASSC->C_or_D = 0;
    oASSC->printMAP = 0;
    oASSC->startTime = 0;
    oASSC->D0 = 1;
    oASSC->CMax = 1e15;
    oASSC->C0 = 1e4;    // mol/m^3

    /*
    --------------------------------------------------------------------------------

    If anybody has a better idea of how to parse inputs please let me know.
    Eventually I'm hoping the GUI will replace a lot of this code.

    --------------------------------------------------------------------------------
    */

    while (std::getline(InputFile, myText))
    {
        sscanf(myText.c_str(), "%s %lf", tempC, &tempD);
        if (strcmp(tempC, "POI:") == 0)
        {
            oASSC->POI = (int)tempD;
        }
        else if (strcmp(tempC, "POI_TH:") == 0)
        {
            oASSC->POI_TH = (int)tempD;
        }
        else if (strcmp(tempC, "POI_DC:") == 0)
        {
            oASSC->POI_DC = tempD;
        }
        else if (strcmp(tempC, "current_density:") == 0)
        {
            oASSC->currentDensity = tempD;
        }
        else if (strcmp(tempC, "startTime:") == 0)
        {
            oASSC->startTime = tempD;
        }
        else if (strcmp(tempC, "stepSize:") == 0)
        {
            oASSC->stepTime = tempD;
        }
        else if (strcmp(tempC, "totalTime:") == 0)
        {
            oASSC->totalTime = tempD;
        }
        else if (strcmp(tempC, "pixelResolution:") == 0)
        {
            oASSC->pixelRes = tempD;
        }
        else if (strcmp(tempC, "C_or_D:") == 0)
        {
            oASSC->C_or_D = (int)tempD;
        }
        else if (strcmp(tempC, "printMaps:") == 0)
        {
            oASSC->printMAP = (int)tempD;
        }
        else if (strcmp(tempC, "CMax:") == 0)
        {
            oASSC->CMax = tempD;
        }
        else if (strcmp(tempC, "Dprime:") == 0)
        {
            oASSC->D0 = tempD;
        }
        else if(strcmp(tempC, "C0:") == 0)
        {
            oASSC->C0 = tempD;
        }
        else if(strcmp(tempC, "PB:") == 0)
        {
            oASSC->PB = (int)tempD;
        }
    }
    return;
}

/*

    Discretization and Setup

*/

void activeSA_2D_ASSC(meshInfo *mesh, ASSCopts *oASSC, char *simData)
{
    /*
        activeSA_2D_ASSC:
        Inputs:
            - pointer to mesh info
            - pointer to ASSC opts
            - pointer to simData array
        Outputs:
            - none.
        
        Function will calculate the active surface area between AM and SE
        particles.
    */
    double SA = 0;
    int row, col;
    for(int i = 0; i < mesh->nElements; i++)
    {
        if(simData[i] != oASSC->POI)
            continue;
        row = i / mesh->numCellsX;
        col = i - row * mesh->numCellsX;

        // check north and south
        
        if (row != 0)
        {
            // check north
            if (simData[i - mesh->numCellsX] == 1)
            {
                SA += 1;
            }
        }

        if(row != mesh->numCellsY - 1)
        {
            // check South
            if (simData[i + mesh->numCellsX] == 1)
            {
                SA += 1;
            }
        }

        // check east and west

        if (col == 0 && oASSC->PB)
        {
            // periodic west
            int tempCol = mesh->numCellsX - 1;
            if(simData[row * mesh->numCellsX + tempCol] == 1)
            {
                SA += 1;
            }
        }
        
        if(col != 0)
        {
            // west
            if (simData[i - 1] == 1)
            {
                SA += 1;
            }
        }

        if (col == mesh->numCellsX - 1 && oASSC->PB)
        {
            // periodic east
            int tempCol = 0;
            if(simData[row*mesh->numCellsX + tempCol] == 1)
            {
                SA += 1;
            }
        }

        if (col != mesh->numCellsX - 1)
        {
            // east
            if(simData[i + 1] == 1)
            {
                SA += 1;
            }
        }
    }// end for

    // calculate SA and SSA
    mesh->numFaces = (long int) SA;
    mesh->SA = SA * mesh->dx * mesh->dy;                // number of faces times face area
    mesh->SSA = (double) mesh->SA / mesh->nElements;    // SA divided by volume

    return;
}

void ASSC_AM_VF(meshInfo *mesh, ASSCopts *oASSC, char *simData)
{
    /*
        ASSC_AM_VF Function:
        Inputs:
            - pointer to meshInfo struct
            - pointer to ASSC options struct
            - pointer to simData
        Outputs:
            - AM_VF
        
        Function will calculate the AM volume fraction for this simulation.
    */

    long int count = 0;

    for(int i = 0; i < mesh->nElements; i++)
    {
        if (simData[i] == oASSC->POI)
            count++;
    }

    oASSC->AM_VF = (double)count/mesh->nElements;

    return;
}

void ASSC_DC(meshInfo *mesh, ASSCopts *oASSC, char *simData, double *DC)
{
    /*
        ASSC_DC Function:
        Inputs:
            - pointer to meshInfo struct
            - pointer to ASSCopts struct
            - pointer to simData array
            - pointer to diffusion coefficient array
        Outputs:
            - none
        
        Function will binarize depending on the POI for the ASSC simulation.
        Only the DC array is modified.
    */

    for(int i = 0; i < mesh->nElements; i++)
    {
        if(simData[i] == oASSC->POI)
        {
            DC[i] = oASSC->POI_DC;
        }
        else
            DC[i] = 0;
    }

    return;
}

void SetBC_ASSC(options *opts, meshInfo *mesh, ASSCopts *oASSC, char *simData, int *BC, double *BC_value)
{
    /*
        Function SetBC_ASSC:
        Inputs:
            - pointer to opts
            - pointer to mesh
            - pointer to ASSCopts (submodel opts)
            - pointer to char simData
            - pointer to BC array
            - pointer to BC value (array)
        Outputs:
            - none
        
        BC Flags:
        0 : No boundary
        1 : Dirichlet
        2 : Neumann
        3 : Robin
        4 : Mixed
        5 : Periodic

        Note: 1, 3, and 4 are not used in this model.
        Function will populate BC_array and BC_value with the appropriate submodel boundary conditions
        and values for the BC's (if applicable).
    */

    // Set some variables to help
    int nCols, nRows;
    nCols = mesh->numCellsX + 2;
    nRows = mesh->numCellsY + 2;

    // Get applied current density

    double Area = mesh->numCellsX * oASSC->pixelRes;

    double appliedCurrent = oASSC->currentDensity * Area;

    /*
        Different Methods to calculate flux:
        
        1: same flux every active surface
        2: distance weighted
        3: ? 
    */

    // Currently only (1) is coded

    double flux;

    flux = mesh->SA/(mesh->dx * mesh->dy);

    oASSC->faceFlux = appliedCurrent / mesh->numFaces;

    // search for boundaries

    int right, left, top, bottom;
    // set col values for right and left
    left = 0;
    right = nCols - 1;

    // set row values for top and bottom
    top = 0;
    bottom = nRows - 1;

    // right and left boundaries (Neumann)

    for (int i = 0; i < nRows; i++)
    {
        // right side
        BC[i * nCols + right] = 5;          // Periodic
        // left side
        BC[i * nCols + left] = 5;           // Periodic
    }

    // top and bottom boundaries

    for (int j = 0; j < nCols; j++)
    {
        // top
        BC[top * nCols + j] = 2;
        BC_value[top * nCols + j] = 0;
        // bottom (have to check it is not AM)

        BC[bottom * nCols + j] = 2;
        // BC_value[bottom * nCols + j] = -appliedCurrent;
        BC_value[bottom * nCols + j] = 0;   // test
    }

    return;
}


void disc2D_ASSC(options     *opts,
                meshInfo    *mesh,
                ASSCopts    *oASSC,
                double      *DC,
                double      *Coeff,
                double      *RHS,
                double      *C0)
{
    /*
        Function disc2D_ASSC:
        Inputs:
            - pointer to options struct
            - pointer to mesh struct
            - pointer to ASSC opts
            - pointer to DC
            - pointer to Coefficient Matrix
            - pointer to RHS
            - pointer to concentration dist. at last time-step
        Outputs:
            - None.
        
        Function will create a 2D + 1D discretization of the given system based on
        central differencing for the space dependent component and Crank-Nicolson
        method for implicit time stepping.
    */

    // Set necessary variables

    int nCols;
    nCols = mesh->numCellsX;

    double dx, dy, dt;
    dx = mesh->dx;
    dy = mesh->dy;
    dt = mesh->dt;

    int row, col;
    long int BC_index;
    double dw, de, ds, dn;

    int tempE, tempW;

    // main loop

    for(int index = 0; index < mesh->nElements; index++)
    {
        // If DC = 0, do not solve
        if(DC[index] == 0)
        {
            Coeff[index*5 + 0] = 1;
            RHS[index] = 0;
            continue;
        }

        // make sure COeff and RHS are 0
        RHS[index] = 0;
        for (int k = 0; k < 5; k++)
        {
            Coeff[index * 5 + k] = 0;
        }

        // Deal with current conditions

        row = index / nCols;
        col = index - row * nCols;

        // get temp indexes in case of periodic BC

        if (col == 0)
        {
            tempW = nCols - 1;
            tempE = col + 1;
        }
        else if (col == nCols - 1)
        {
            tempW = col - 1;
            tempE = 0;
        }
        else
        {
            tempW = col - 1;
            tempE = col + 1;
        }

        /*
            Indexing for coeff marix:

            0 : P       i
            1 : W       i - 1
            2 : E       i + 1
            3 : S       i + nCols
            4 : N       i - nCols
        */

        // RHS

        RHS[index] += 2.0 * (dx * dy)/dt * C0[index];

        // West

        if(DC[row * nCols + tempW] == 0)
        {
            RHS[index] += -oASSC->faceFlux;
        }
        else
        {
            dw = WeightedHarmonicMean(dx/2, dx/2, DC[index], DC[row * nCols + tempW]);
            Coeff[index*5 + 1] = -dw * dx/dy;
            Coeff[index*5 + 0] += dw * dx/dy;
            RHS[index] -= Coeff[index*5 + 1] * C0[row * nCols + tempW];
        }

        // East

        if(DC[row * nCols + tempE] == 0)
        {
            RHS[index] += oASSC->faceFlux;
        }
        else
        {
            de = WeightedHarmonicMean(dx/2, dx/2, DC[index], DC[row * nCols + tempE]);
            Coeff[index*5 + 2] = -de * dx/dy;
            Coeff[index*5 + 0] += de * dx/dy;
            RHS[index] -= Coeff[index*5 + 2] * C0[row * nCols + tempE];
        }

        // South

        if (row != mesh->numCellsY - 1)
        {
            if(DC[(row + 1) * nCols + col] == 0)
            {
                RHS[index] += -oASSC->faceFlux;
            }
            else
            {
                ds = WeightedHarmonicMean(dy/2, dy/2, DC[index], DC[(row + 1) * nCols + col]);
                Coeff[index * 5 + 3] = -ds * dy / dx;
                Coeff[index * 5 + 0] += ds * dy/dx;
                RHS[index] -= Coeff[index*5 + 3] * C0[(row + 1) * nCols + col]; 
            }
        }

        // North

        if(row != 0)
        {
            if(DC[(row - 1) * nCols + col] == 0)
            {
                RHS[index] += oASSC->faceFlux;
            }
            else
            {
                dn = WeightedHarmonicMean(dy/2, dy/2, DC[(row - 1) * nCols + col], DC[index]);
                Coeff[index * 5 + 4] = -dn * dy/dx;
                Coeff[index * 5 + 0] += dn * dy/dx;
                RHS[index] -= Coeff[index * 5 + 4] * C0[(row - 1) * nCols + col];
            }
        }

        // P contribution from last time step

        RHS[index] += -Coeff[index*5 + 0] * C0[index];
        Coeff[index*5 + 0] += 2.0 * dx * dy / dt;

        //end
    }

    return;
}

// Test function below

void test_funct(void)
{
    printf("hello world\n");

    return;
}


#endif