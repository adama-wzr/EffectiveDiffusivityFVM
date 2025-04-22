#ifndef _TSSD
#define _TSSD

#include <math.h>
#include <fstream>
#include <iostream>
#include <string>

#include <helper.cuh>
#include <datastructures.cpp>

/*

Handling user input in TSSD submodel:

*/

void printTSSD(options *opts, TSSDopts *oTSSD)
{
    /*
        print user options
    */

    printf("--------------------------------------------\n\n");
    printf("            TSSD Model Inputs:              \n\n");
    printf("--------------------------------------------\n\n");

    printf("Input image name: %s\n", opts->inputFilename);
    printf("Number of Phases: %d\n", opts->numDC);
    for(int i = 0; i < opts->numDC; i++)
    {
        printf("Phase = %d\n", i+1);
        printf("Threshold (Upper Bound) = %d\n", opts->DC_TH[i]);
        if(i == oTSSD->POI - 1)
            printf("DC[%d] = ???\n", i+1);
        else
        {
            printf("DC[%d] = %1.3e m^2/s\n", i+1, opts->DC[i]);
        }
    }

    printf("POI: %d\n", oTSSD->POI);

    // mesh amplificaiton

    printf("Mesh Refine X = %d\n", opts->MeshIncreaseX);
    printf("Mesh Refine Y = %d\n", opts->MeshIncreaseY);
    if (opts->nD == 3)
    {
        printf("Mesh Refine Z = %d\n", opts->MeshIncreaseZ);
    }

    // convergence

    printf("Max. Iterations: %ld\n", opts->MAX_ITER);
    printf("Convergence: %1.3e\n", opts->ConvergeCriteria);

    if (opts->useGPU == 1)
    {
        printf("Using %d GPU(s)\n", opts->nGPU);
    }
    else
    {
        printf("Number of Threads = %d\n", opts->nThreads);
    }

    if(oTSSD->printMAP)
        printf("Printing CMaps and FMaps\n");
    else
        printf("Not printing maps, only save C(y,t)\n");


    // TSSD Specific Options
    printf("--------------------------------------------\n\n");
    printf("            DC Search Options               \n\n");
    printf("--------------------------------------------\n\n");

    printf("DC Max:  %1.3e\n", oTSSD->DC_Max);
    printf("DC Min:  %1.3e\n", oTSSD->DC_Min);
    printf("DC Step: %1.3e\n", oTSSD->DC_Step);

    printf("Start Time: %1.3f (sec)\n", oTSSD->startTime);
    printf("Stop Time: %1.3f (sec)\n", oTSSD->totalTime);
    printf("Save Interval: %1.3f (sec)\n", oTSSD->stepSize);

    if(oTSSD->C_or_D)
        printf("Simulating Charge\n");
    else
        printf("Simulating Discharge Step\n");
    
    printf("Current Density: %1.3f A\n", oTSSD->current_density);

    if(oTSSD->D0 == 1)
    {
        printf("Anomalous Diffusion Information not entered.\n");
    }
    else
    {
        printf("Trace Species Diffusion: %1.3e m^2/s\n", oTSSD->D0);
        printf("CMax: %1.3e mol/m^3\n", oTSSD->CMax);
    }

    return;
}

void readInputTSSD(char *FileName, TSSDopts *oTSSD)
{

    /*
        readInputTSSD Function:
        Inputs:
            - FileName: pointer to where the input file name is stored.
            - TSSDopts: data structure options for TSSD simulation
        Outputs: None

        Function reads the input file for TSSD.
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

    oTSSD->C_or_D = 0;
    oTSSD->printMAP = 0;
    oTSSD->startTime = 0;
    oTSSD->D0 = 1;
    oTSSD->CMax = 1e15;
    oTSSD->DC_Min = 1e-15; // m^2/s
    oTSSD->DC_Max = 1e-10; // m^2/s

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
            oTSSD->POI = (int)tempD;
        }
        else if (strcmp(tempC, "DC_Max:") == 0)
        {
            oTSSD->DC_Max = tempD;
        }
        else if (strcmp(tempC, "DC_Min:") == 0)
        {
            oTSSD->DC_Min = tempD;
        }
        else if (strcmp(tempC, "DC_Step:") == 0)
        {
            oTSSD->DC_Step = tempD;
        }
        else if (strcmp(tempC, "current_density:") == 0)
        {
            oTSSD->current_density = tempD;
        }
        else if (strcmp(tempC, "startTime:") == 0)
        {
            oTSSD->startTime = tempD;
        }
        else if (strcmp(tempC, "stepSize:") == 0)
        {
            oTSSD->stepSize = tempD;
        }
        else if (strcmp(tempC, "totalTime:") == 0)
        {
            oTSSD->totalTime = tempD;
        }
        else if (strcmp(tempC, "C_or_D:") == 0)
        {
            oTSSD->C_or_D = (int)tempD;
        }
        else if (strcmp(tempC, "printMaps:") == 0)
        {
            oTSSD->printMAP = (int)tempD;
        }
    }
    return;
}

#endif