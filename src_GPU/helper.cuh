/*

License information:

Upcoming.


Code contributors:

Andre Adam. 

Last Update: 
01/16/2025

*/





#ifndef _HELPER
#define _HELPER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <vector>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdbool.h>
#include <fstream>
#include <cfloat>
#include <set>
#include <string>
#include "cuda_runtime.h"
#include "cuda.h"

typedef struct
{
	double *DC;                 // array with diffusion coefficients
    unsigned char *DC_TH;       // Upper limit threshold for phase differentiation when reading jpg's
    int numDC;               // number of diffusion coefficients
	int MeshIncreaseX;		    // Mesh refinement in x-direction
	int MeshIncreaseY;		    // Mesh refinement in y-direction
    int MeshIncreaseZ;          // Mesh refinement in the z-direction
	double CLeft;			    // Concentration of trace species in left boundary
	double CRight;			    // Concentration of trace species in right boundary
	long int MAX_ITER;		    // Max iterations
	double ConvergeCriteria;    // Convergence Criteria
	char *inputFilename;	    // Input filename
	char *outputFilename;	    // Output filename
	int printCmap;			    // print concentration map (true/false) flag
	char *CMapName;			    // Concentration map name
	int verbose;			    // verbose flag
	int BatchFlag;			    // Batch flag
	int NumImgBatch;		    // Number of images in the batch
    char SteadyStateFlag;       // steady state simulation or time dependent ?
    int height;                 // height in number of pixels
    int width;                  // width in number of pixels
    int depth;                  // depth in number of pixels
    double nD;                  // number of dimensions
    char inputType;             // Input format for 3D simulations (0 default .csv, 1 is stack)
} options;

typedef struct
{
	int Width;
	int Height;
    int Depth;
	double time;
	unsigned char *target_data;
	double deff;
	bool PathFlag;
	double conv;
} simulationInfo;

typedef struct
{
	int numCellsX;
	int numCellsY;
	int nElements;
	double dx;
	double dy;
} meshInfo;


int printOptions(options* opts)
{
    /*
		Function printOptions:
        Inputs:
            - pointer to opts struct
        Outputs:
            - None
        
        The function is only called when Verbose = true. It will
        print the user entered options to the command line. For saving cmd output
        into a text file, please do so externally.
	*/

    printf("--------------------------------------\n\n");
    printf("Current selected options:\n\n");
    printf("--------------------------------------\n");
    printf("Number of Dimensions: %d\n", opts->nD);
    if(opts->BatchFlag)
    {
        printf("Running a bacth of size = %d\n", opts->NumImgBatch);
    } else
    {
        if(opts->inputType == 0)
        {
            printf("Input Method = csv\n");
            printf("Filename = %s\n", opts->inputFilename);
            printf("Structure Width  = %d\n", opts->width);
            printf("Structure Height = %d\n", opts->height);
            printf("Structure Depth  = %d\n", opts->depth);

            // If nD = 2 make sure to set input type to 2
        
        
        }
        else if(opts->inputType == 1)
        {
            printf("Input Method = jpg stack\n");
            printf("Stack Size = %d\n", opts->depth);
        }

        // stopped here
        
    }
    
    printf("Number of phases:\n");

	// if(opts->BatchFlag == 0){
	// 	printf("--------------------------------------\n\n");
	// 	printf("Current selected options:\n\n");
	// 	printf("--------------------------------------\n");
	// 	printf("Number of Phases = %d\n", opts->nPhase);
	// 	printf("DC Fluid = %1.3e\n", opts->DCfluid);
	// 	printf("DC Solid = %1.3e\n", opts->DCsolid);
	// 	printf("DC Gas = %1.3e\n", opts->DCgas);
	// 	printf("Concentration Left = %.2f\n", opts->CLeft);
	// 	printf("Concentration Right = %.2f\n", opts->CRight);
	// 	printf("Mesh Amp. X = %d\n", opts->MeshIncreaseX);
	// 	printf("Mesh Amp. Y = %d\n", opts->MeshIncreaseY);
	// 	printf("Maximum Iterations = %ld\n", opts->MAX_ITER);
	// 	printf("Convergence = %.10f\n", opts->ConvergeCriteria);
	// 	printf("Name of input image: %s\n", opts->inputFilename);
	// 	printf("Name of output file: %s\n", opts->outputFilename);

	// 	if(opts->printCmap == 0){
	// 		printf("Print Concentration Map = False\n");
	// 	} else{
	// 		printf("Concentration Map Name = %s\n", opts->CMapName);
	// 	}
	// 	printf("--------------------------------------\n\n");
	// } else if(opts->BatchFlag == 1){
	// 	printf("--------------------------------------\n\n");
	// 	printf("Running Image Batch:\n\n");
	// 	printf("Number of Phases = %d\n", opts->nPhase);
	// 	printf("DC Fluid = %1.3e\n", opts->DCfluid);
	// 	printf("DC Solid = %1.3e\n", opts->DCsolid);
	// 	printf("DC Gas = %1.3e\n", opts->DCgas);
	// 	printf("Concentration Left = %.2f\n", opts->CLeft);
	// 	printf("Concentration Right = %.2f\n", opts->CRight);
	// 	printf("Mesh Amp. X = %d\n", opts->MeshIncreaseX);
	// 	printf("Mesh Amp. Y = %d\n", opts->MeshIncreaseY);
	// 	printf("Maximum Iterations = %ld\n", opts->MAX_ITER);
	// 	printf("Convergence = %.10f\n", opts->ConvergeCriteria);
	// 	printf("Name of output file: %s\n", opts->outputFilename);
	// 	printf("Number of files to run: %d\n", opts->NumImg);
	// 	if (opts->printCmap == 1){
	// 		printf("Printing Concentration Distribution for all images.\n");
	// 	} else{
	// 		printf("No Concentration maps will be printed.\n");
	// 	}
	// 	printf("--------------------------------------\n\n");
	// }
    return 0;
}


void readInputGeneral(char* FileName, options* opts){

	/*
		readInputGeneral Function:
		Inputs:
			- FileName: pointer to where the input file name is stored.
			- struct options: pass a struct with the options.
		Outputs: None

		Function reads the input file and stores the options in the opts struct.
	*/

    // initiate necessary variables for input reading
    std::string myText;
    
    char tempC[1000];
	double tempD;
	char tempFilenames[1000];
	std::ifstream InputFile(FileName);

    // initiate arrays for storing names of input/output files

    opts->inputFilename  =   (char *)malloc(1000*sizeof(char));
    opts->outputFilename =   (char*)malloc(1000*sizeof(char));
	opts->CMapName       =   (char*)malloc(1000*sizeof(char));
    
    // variables for reading diffusion coefficients (DC) and the thresholds (DC_TH)
    
    opts->numDC = 0;
    char tempDC[20];
    char tempDC_TH[20];
    int DC_read = 0;
    int DC_TH_read = 0;

    // Default values set here

    opts->MeshIncreaseX = 1;
    opts->MeshIncreaseY = 1;
    opts->MeshIncreaseZ = 1;

    /*
    --------------------------------------------------------------------------------

    If anybody has a better idea of how to parse inputs please let me know.
    Eventually I'm hoping the GUI will replace a lot of this code.

    --------------------------------------------------------------------------------
    */


    while(std::getline(InputFile, myText))
    {
        sscanf(myText.c_str(), "%s %lf", tempC, &tempD);
        if(strcmp(tempC, "nD:") == 0)
        {
            opts->nD = tempD;
        }
        else if(strcmp(tempC, "numDC:") == 0)
        {
            opts->numDC = (int)tempD;
            // allocate the space in memory
            opts->DC = (double *)malloc(opts->numDC*sizeof(double));
            opts->DC_TH = (unsigned char *)malloc(opts->numDC*sizeof(char));
            // set memory
            memset(opts->DC, 0, opts->numDC*sizeof(double));
            memset(opts->DC_TH, 0, opts->numDC*sizeof(char));
            DC_read++;
            DC_TH_read++;
        }
        else if(strcmp(tempC, tempDC) == 0)
        {
            opts->DC[DC_read - 1] = tempD;
            DC_read++;
        }
        else if(strcmp(tempC, tempDC_TH) == 0)
        {
            opts->DC_TH[DC_TH_read - 1] = (unsigned char)tempD;
            DC_TH_read++;
        }
        else if(strcmp(tempC, "MeshAmpX:") == 0)
        {
            opts->MeshIncreaseX = (int)tempD;
        }
        else if(strcmp(tempC, "MeshAmpY:") == 0)
        {
            opts->MeshIncreaseY = (int)tempD;
        }
        else if(strcmp(tempC, "MeshAmpZ:") == 0)
        {
            opts->MeshIncreaseZ = (int)tempD;
        }
        else if(strcmp(tempC, "InputName:") == 0)
        {
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
	 		strcpy(opts->inputFilename, tempFilenames);
        }
        else if(strcmp(tempC, "OutputName:") == 0)
        {
	 		sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
	 		strcpy(opts->outputFilename, tempFilenames);
        }
        else if(strcmp(tempC, "printCMap:") == 0){
	 		opts->printCmap = (int)tempD;
	 	}
        else if(strcmp(tempC, "CMapName:") == 0)
        {
	 		sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
	 		strcpy(opts->CMapName, tempFilenames);
	 	}
        else if(strcmp(tempC, "Convergence:") == 0)
        {
	 		opts->ConvergeCriteria = tempD;
	 	}
        else if(strcmp(tempC, "MaxIter:") == 0)
        {
	 		opts->MAX_ITER = (long int)tempD;
	 	}
        else if(strcmp(tempC, "Verbose:") == 0)
        {
	 		opts->verbose = (int)tempD;
	 	}
        else if(strcmp(tempC, "RunBatch:") == 0)
        {
	 		opts->BatchFlag = (int)tempD;
	 	}
        else if(strcmp(tempC, "NumImages:") == 0)
        {
	 		opts->NumImgBatch = (int)tempD;
        }
        else if(strcmp(tempC, "CL:") == 0)
        {
            opts->CLeft = tempD;
        }
        else if(strcmp(tempC, "CR:") == 0)
        {
            opts->CRight = tempD;
        }
        else if(strcmp(tempC, "SS:") == 0)
        {
            opts->SteadyStateFlag = (char)tempD;
        }
        else if(strcmp(tempC, "width:") == 0)
        {
            opts->width = (int)tempD;
        }
        else if(strcmp(tempC, "height:") == 0)
        {
            opts->height = (int)tempD;
        }
        else if(strcmp(tempC, "depth:") == 0)
        {
            opts->depth = (int)tempD;
        }
        else if(strcmp(tempC, "inputType"))
        {
            opts->inputType = (char)tempD;
        }

        // Update the number of expected diffusion coefficients and thresholding for image
        // processing

        if (DC_read <= opts->numDC)     sprintf(tempDC, "D%d:", DC_read);
        if (DC_TH_read <= opts->numDC)  sprintf(tempDC_TH, "D_TH%d:", DC_TH_read);
    }
    return;
}

#endif