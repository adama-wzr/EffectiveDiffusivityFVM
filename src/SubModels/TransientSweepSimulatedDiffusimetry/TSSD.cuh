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
    printf("POI: %d\n", oTSSD->POI);


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
            oTSSD->C_or_D = (int)tempD;
        }
    }
    return;
}

#endif