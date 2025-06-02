#ifndef _MIG
#define _MIG

#include <math.h>
#include <fstream>
#include <iostream>
#include <string>

#include <datastructures.cpp>
#include <helper.cuh>
#include <constants.cpp>


/*

    Read Input

*/

int readInputMig(Migration *mig)
{
    /*
        Function readInputMig:
        Inputs:
            - Migration datastructure (migration options array)
        Outputs:
            - none
        
        Function will read the input file with configurations
        into the migration struct.
    */
    
    // try to open the file
    
    char FileName[100];

    sprintf(FileName, "inputMig.txt");

    std::string myText;

    char tempC[1000];
    double tempD;
    char tempFilenames[1000];

    std::ifstream InputFile(FileName);

    if(InputFile.fail())
    {
        printf("Failed to read file %s\n", FileName);
        return 1;
    }

    // set defaults
    for(int i = 0; i < 3; i++)
        mig->dE_dL[i] = 0.0f;
    
    mig->T = 298.0;       // K

    // read file

    while (std::getline(InputFile, myText))
    {
        sscanf(myText.c_str(), "%s %lf", tempC, &tempD);
        if (strcmp(tempC, "T:") == 0)
        {
            mig->T = tempD;
        }
        else if(strcmp(tempC, "dE_dx:") == 0)
        {
            mig->dE_dL[0] = tempD;
        }
        else if(strcmp(tempC, "dE_dy:") == 0)
        {
            mig->dE_dL[1] = tempD;
        }
        else if(strcmp(tempC, "dE_dz:") == 0)
        {
            mig->dE_dL[2] = tempD;
        }
    }
    
    return 0;
}


/*

    2D Discretization

*/

void Disc_Mob2D(double *Coeff, double *RHS, meshInfo *mesh)
{
    
    return;
}


#endif