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

void Disc_Mig2D(double *Coeff, double *DC, double *RHS, double *C0, options *opts, meshInfo *mesh, Migration *mig)
{
    /*
        Function Disc_Mig2D:
        Inputs:
            - pointer to coefficient matrix
            - pointer to DC array
            - pointer to RHS array
            - pointer to (previous time-step) concentration array
            - pointer to opts struct
            - pointer to mesh struct
            - pointer to migration struct
        Outputs:
            - none.
        
        This function will provide the migration contribution to the diffusion 
        of charged species. 
    */


    // declare needed variables
    double ap;
    double dn, ds;
    double dx = mesh->dx;
    double dy = mesh->dy;
    int row, col;

    FILE *TEST;

    TEST = fopen("TestDisc.csv", "w+");
    fprintf(TEST, "ap,Mig\n");

    for (int i = 0; i < mesh->nElements; i++)
    {
        // get row and column

        row = i/mesh->numCellsX;
        col = i - row * mesh->numCellsX;

        // Check north

        if(row == 0)
        {
            // no North
            dn = DC[i];
        }else
        {
            // yes North
            dn = WeightedHarmonicMean(dy/2, dy/2, DC[i], DC[i - 1]);
            if (DC[i] == 0)
                dn = 0;
        }

        if(row == mesh->numCellsY - 1)
        {
            // no South
            ds = DC[i];
        }else
        {
            // yes South
            ds = WeightedHarmonicMean(dy/2, dy/2, DC[i], DC[i + 1]);
        }

        fprintf(TEST, "%1.3e,%1.3e\n", Coeff[i*5 + 0], opts->charge*FARADAY/(GAS_C * mig->T)*mig->dE_dL[1]*(dn - ds));

        // update coefficient

        Coeff[i*5 + 0] += opts->charge*FARADAY/(GAS_C * mig->T)*mig->dE_dL[1]*(dn - ds);
        RHS[i] += -opts->charge*FARADAY/(GAS_C * mig->T)*mig->dE_dL[1]*(dn - ds)*C0[i];
    }

    fclose(TEST);
    
    return;
}


#endif