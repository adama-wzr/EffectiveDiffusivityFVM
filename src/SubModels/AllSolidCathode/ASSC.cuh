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

void printInputASSC(options *opts, ASSCopts *oASSC)
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


// Test function below

void test_funct(void)
{
    printf("hello world\n");

    return;
}


#endif