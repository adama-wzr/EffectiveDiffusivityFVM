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
    int printOut;               // Flag to print output or not
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
    int nD;                  // number of dimensions
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
    int numCellsZ;
	long int nElements;
	double dx;
	double dy;
    double dz;
} meshInfo;

// Define coords for Flood Fill

typedef std::tuple<int, int, int> coord;


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
    printf("InputType = %d\n", opts->inputType);
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
            printf("Diffusion Coefficients:\n");
            for(int i = 0; i < opts->numDC; i++)
            {
                printf("D%d = %1.3e\n",i+1, opts->DC[i]);
            }

            // If nD = 2 make sure to set input type to 2
        }
        else if(opts->inputType == 1)
        {
            printf("Input Method = jpg stack\n");
            printf("Stack Size = %d\n", opts->depth);
            printf("Diffusion Coefficients and Processing Thresholds:\n");
            for(int i = 0; i < opts->numDC; i++)
            {
                printf("D%d = %1.3e\n",i+1, opts->DC[i]);
                printf("D_TH%d = %d\n", i+1, opts->DC_TH[i]);
            }
        }
        else if(opts->inputType == 2)
        {
            printf("Filename = %s\n",opts->inputFilename);
            printf("Diffusion Coefficients and Processing Thresholds:\n");
            for(int i = 0; i < opts->numDC; i++)
            {
                printf("D%d = %1.3e\n",i+1, opts->DC[i]);
                printf("D_TH%d = %d\n", i+1, opts->DC_TH[i]);
            }
        }
        // check steady-state flag
        if(opts->SteadyStateFlag == 1)
        {
            printf("Steady-State Simulation Mode Selected\n");
        }
        else if(opts->SteadyStateFlag == 0)
        {
            printf("Time-dependent Simulation Selected\n");
            printf("Time discretization:\n");
            /*
                Add here different time discretization requirements
            */
        }
        printf("Mesh Refine X = %d\n", opts->MeshIncreaseX);
        printf("Mesh Refine Y = %d\n", opts->MeshIncreaseY);
        if (opts->nD == 3)
        {
            printf("Mesh Refine Z = %d\n", opts->MeshIncreaseZ);
        }

        printf("Max. Iterations: %ld\n", opts->MAX_ITER);
        printf("Convergence: %1.3e\n", opts->ConvergeCriteria);

        if(opts->printCmap == 1)
        {
            printf("CMAP Name: %s\n", opts->CMapName);
        }
        if(opts->printOut == 1)
        {
            printf("Output File Name: %s\n", opts->outputFilename);
        }
    }
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

    opts->BatchFlag = 0;
    opts->inputType = 0;

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
            opts->nD = (int)tempD;
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
        else if(strcmp(tempC, "inputType:") == 0)
        {
            opts->inputType = (char)tempD;
        }
        else if(strcmp(tempC, "printOutput:") == 0)
        {
            opts->printOut = (int)tempD;
        }

        // Update the number of expected diffusion coefficients and thresholding for image
        // processing

        if (DC_read <= opts->numDC)     sprintf(tempDC, "D%d:", DC_read);
        if (DC_TH_read <= opts->numDC)  sprintf(tempDC_TH, "D_TH%d:", DC_TH_read);
    }
    return;
}

int readCSV3D(options* opts, char* simObject)
{
    /*
        Function readCSSV3D:
        Inputs:
            - pointer to options data structure
            - pointer to simObject array, where the structure will be saved
                with the appropriate flags.
        Output:
            - None
        
        The function will populate the simObject array according to the data in the
        input .csv file.

    */
    // read structure
    int height, width, depth;
    long int nElements;

    height = opts->height;
    width = opts->width;
    depth = opts->depth;
    nElements = height*width*depth;

    // declare arrays to hold coordinates for all specified phases

    int *x = (int *)malloc(sizeof(int)*nElements);
    int *y = (int *)malloc(sizeof(int)*nElements);
    int *z = (int *)malloc(sizeof(int)*nElements);
    int *phase = (int *)malloc(sizeof(int)*nElements);

    // Read structure file

    FILE *target_data;

    target_data = fopen(opts->inputFilename, "r");

    // check if file exists

    if (target_data == NULL){
        fprintf(stderr, "Error reading file. Exiting program.\n");
        return 1;
    }

    char header[20];

    fscanf(target_data, "%c,%c,%c,%s", &header[0], &header[1], &header[2], &header[3]);

    // if (opts->verbose) printf("Header = %s\n", header);      // debug mainly

    // read coordinates from input file

    size_t count = 0;

    while (fscanf(target_data, "%d,%d,%d,%d", &x[count], &y[count], &z[count], &phase[count]) == 4)
    {
        count++;
    }

    long int index = 0;

    for(long int i = 0; i<count; i++)
    {
        index = z[i]*height*width + y[i]*width + x[i];
        simObject[index] = phase[i];    // the diffusivities later are assigned based on this number
    }

    // memory management

    free(x);
    free(y);
    free(z);
    free(phase);

    return 0;
}

int SetDC3D(options* opts, meshInfo* mesh, double* DC, char* simObject)
{
    /*
        Function SetDC3D:
        Inputs:
            - pointer to options struct
            - pointer to mesh struct
            - pointer to DC, an array where the diffusion coefficients will be stored
        Outputs:
            - None.
        The function will set the diffusion coefficient of the grid. 
    */

    for(int k = 0; k<mesh->numCellsZ; k++)
    {
        for(int i = 0; i<mesh->numCellsY; i++)
        {
            for(int j = 0; j<mesh->numCellsX; j++)
            {
                // index for original array
                int targetRow = i / opts->MeshIncreaseY;
                int targetCol = j / opts->MeshIncreaseX;
                int targetSlice = k / opts->MeshIncreaseZ;
                int targetIndex = targetSlice*opts->height*opts->width 
                            + targetRow*opts->width + targetCol;
                // index for array with meshAmp
                int index = k*mesh->numCellsX*mesh->numCellsY
                            +i*mesh->numCellsX + j;
                // Identify phase and diffusion coefficient
                int localPhase = simObject[targetIndex];
                double localDC = opts->DC[localPhase];
                // Store data
                DC[index] = localDC;
            }
        }
    }

    return 0;
}

int SetBC_DeffSetup3D(options* opts, meshInfo* mesh, char* BC, double* BC_Value)
{
    /*
        Function SetBC_DeffSetup:
        Inputs:
            - pointer to options struct
            - pointer to mesh struct
            - pointer to BC classification array
            - pointer to BC_value array (value of BC for Neumann or Dirichlet)
        Output:
            - None
        
        The function will classify the entire BC array with 2 Dirichlet conditions on the left
        and right, while all other boundaries will be set to zero flux Neumann boundaries.
    */
    // Set some variables to help
    int nCols, nRows, nSlices;
    nCols = mesh->numCellsX + 2;
    nRows = mesh->numCellsY + 2;
    nSlices = mesh->numCellsZ + 2;
    // On the BC array, we need to classify right and left as Dirichlet, and assign the values
    // on BC_Value
    // All the Neumann condition have to be assigned on BC, but no change in BC_Value is required.

    int right, left, top, bottom, front, back;

    left = 0;
    right = mesh->numCellsX + 1;

    top = 0;
    bottom = mesh->numCellsY + 1;

    front = 0;
    back = mesh->numCellsZ + 1;

    // right and left boundaries (Dirichlet)

    for(int row = 0; row < nRows; row++)
    {
        for(int slice = 0; slice < nSlices; slice++)
        {
            long int index1 = slice*nRows*nCols + row*nRows + left;
            long int index2 = slice*nRows*nCols + row*nRows + right;
            // Left boundary
            BC[index1] = 1;         // Dirichlet flag
            BC_Value[index1] = opts->CLeft;
            // Rigth boundary
            BC[index2] = 1;
            BC_Value[index2] = opts->CRight;
        }
    }

    // Top and Bottom boundaries (Neumann)

    for(int slice = 0; slice < nSlices; slice++)
    {
        for(int col = 0; col < nCols; col++)
        {
            long int index1 = slice*nRows*nCols + top*nCols + col;
            long int index2 = slice*nRows*nCols + bottom*nCols + col;
            // Top
            BC[index1] = 2;
            // Bottom
            BC[index2] = 2;
        }
    }

    // Back and Front boundaries (Neumann)

    for(int row = 0; row < nRows; row++)
    {
        for(int col = 0; col < nCols; col++)
        {
            int index1 = front*nRows*nCols + row*nCols + col;
            int index2 = back*nRows*nCols + row*nCols + col;
            // Front
            BC[index1] = 2;
            // Back
            BC[index2] = 2;
        }
    }

    return 0;
}

int FloodFill3D_DeffSetup(meshInfo* mesh, char* BC, double* DC)
{
    /*
        FloddFill3D_DeffSetup function:
        Inputs:
            - pointer to mesh struct
            - pointer to array with BC's
            - pointer to array with DC's
        Outputs:
            - None
        
        The function will search the domain, and will set all DC values that are too
        low to a Neumann BC with zero flux. Non-participating media will also be flagged
        accordingly.
    */

    char* Domain = (char *)malloc(mesh->nElements*sizeof(char));

    // Initialize all the impermeable matter in the domain:

    for(long int index = 0; index < mesh->nElements; index++)
    {
        int slice = index/(mesh->numCellsX*mesh->numCellsY);
        int row = (index - slice*mesh->numCellsX*mesh->numCellsY);
        int col = (index - slice*mesh->numCellsX*mesh->numCellsY - row*mesh->numCellsX);
        int indexBC = (slice + 1)*mesh->numCellsX*mesh->numCellsY + (row + 1)*mesh->numCellsX + (col + 1);
        if(DC[index] < 1e-15)
        {
            Domain[index] = 1;
            BC[indexBC] = 2;    // set BC to Neumann
        } else
        {
            Domain[index] = -1;
        }
    }

    // Find Fluid in both boundaries, add to open list

    std::set<coord> cList;

    int left = 0;
    int right = mesh->numCellsX;

    for(int row = 0; row < mesh->numCellsY; row++)
    {
        for(int slice = 0; slice < mesh->numCellsZ; slice++)
        {
            long int indexL = slice*mesh->numCellsX*mesh->numCellsY + row*mesh->numCellsX + left;
            long int indexR = slice*mesh->numCellsX*mesh->numCellsY + row*mesh->numCellsX + right;
            // set left
            if(Domain[indexL] == -1)
            {
                Domain[indexL] = 0;
                cList.insert(std::tuple(left, row, slice));
            }
            if(Domain[indexR] == -1)
            {
                Domain[indexR] = 0;
                cList.insert(std::tuple(right, row, slice));
            }
        }
    }

    // Search Full Domain

    while(!cList.empty())
    {
        // pop first item on the list
        coord pop = *cList.begin();

        // remove from open list
        cList.erase(cList.begin());

        // get coordinates from the list
        int col     = std::get<0>(pop);
        int row     = std::get<1>(pop);
        int slice   = std::get<2>(pop);

        /*
            We need to check North, South, East, and West for more fluid:
            
            North = col + 0, row - 1, slice + 0
            South = col + 0, row + 1, slice + 0
            East  = col + 1, row + 0, slice + 0
            West  = col - 1, row + 0, slice + 0
            Front = col + 0, row + 0, slice - 1
            Back  = col + 0, row + 0, slice + 1
        
            Note that diagonals are not considered a connection.
            This code assumes no periodic boundary conditions (currently).
        */

        int tempRow, tempCol, tempSlice;
        long int tempIndex;
        
        // North

        tempCol = col;
        tempSlice = slice;

        if(row != 0)
        {
            tempRow = row - 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // South

        if(row != mesh->numCellsY - 1)
        {
            tempRow = row + 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // Front

        tempCol = col;
        tempRow = row;

        if(slice != 0)
        {
            tempSlice = slice - 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // Back

        if(slice != mesh->numCellsZ - 1)
        {
            tempSlice = slice + 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // West

        tempRow = row;
        tempSlice = slice;

        if(col != 0)
        {
            tempCol = col - 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }

        // East

        if(col != mesh->numCellsX - 1)
        {
            tempCol = col + 1;
            tempIndex = tempSlice*mesh->numCellsX*mesh->numCellsY 
                        + tempRow*mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::tuple(tempCol, tempRow, tempSlice));
            }
        }
        // repeat until cList is empty
    }

    // Every flag that is still -1 means a non-participating media

    for(int index = 0; index<mesh->nElements; index++)
    {
        if(Domain[index] != -1) continue;

        int slice = index/(mesh->numCellsX*mesh->numCellsY);
        int row = (index - slice*mesh->numCellsX*mesh->numCellsY);
        int col = (index - slice*mesh->numCellsX*mesh->numCellsY - row*mesh->numCellsX);
        int indexBC = (slice + 1)*mesh->numCellsX*mesh->numCellsY + (row + 1)*mesh->numCellsX + (col + 1);

        BC[indexBC] = -1;
    }
    
    // memory management
    free(Domain);

    return 0;

}

int SteadyStateSim3D(options* opts)
{
    /*
        Function SteadyStateSim3D:
        Inputs:
            - pointer to options data structure
        Outputs:
            - None.
        
        Function will control the entire simulation of a 3D structure in Steady-State
        operation.
    */

    // Initialize simulation data structures
    meshInfo mesh;

    simulationInfo simInfo;

    // populate mesh info with available information

    mesh.numCellsX = opts->width*opts->MeshIncreaseX;
    mesh.numCellsY = opts->height*opts->MeshIncreaseY;
    mesh.numCellsZ = opts->depth*opts->MeshIncreaseZ;
    
    mesh.nElements = mesh.numCellsX*mesh.numCellsY*mesh.numCellsZ;

    mesh.dx = (double)1.0/mesh.numCellsX;
    mesh.dy = (double)1.0/mesh.numCellsY;
    mesh.dz = (double)1.0/mesh.numCellsZ;

    // Read structure
    
    char* simObject = (char *)malloc(opts->height*opts->width*opts->depth*sizeof(char));
    
    memset(simObject, 0, opts->height*opts->width*opts->depth*sizeof(char));    // initialized to pore-space

    readCSV3D(opts, simObject);

    // Declare and define BC's and DC's for the domain

    double *DC = (double *)malloc(sizeof(double)*mesh.nElements);
    char *BC   = (char *)  malloc(sizeof(char)*(mesh.numCellsX + 2) * 
                            (mesh.numCellsX + 2) * (mesh.numCellsX + 2));
    double *BC_Value = (double *)malloc(sizeof(double)*(mesh.numCellsX + 2) * 
                            (mesh.numCellsX + 2) * (mesh.numCellsX + 2));
    
    memset(DC, 0, mesh.nElements*sizeof(double));
    memset(BC, 0, (mesh.numCellsX + 2)*(mesh.numCellsY + 2)
                    *(mesh.numCellsZ + 2)*sizeof(char));
    memset(BC_Value, 0, (mesh.numCellsX + 2)*(mesh.numCellsY + 2)
                    *(mesh.numCellsZ + 2)*sizeof(double));

    // note BC array has space for ``ghost'' grid boundaries

    // Set DC's

    SetDC3D(opts, &mesh, DC, simObject);

    // Set BC's

    SetBC_DeffSetup3D(opts, &mesh, BC, BC_Value);
    
    // set if D[i,,j,k] < 10^-15, D[i,j,k] = 0 becomes a Neumann BC

    for(int index = 0; index<mesh.nElements; index++)
    {
        int slice = index/(mesh.numCellsX*mesh.numCellsY);
        int row = (index - slice*mesh.numCellsX*mesh.numCellsY);
        int col = (index - slice*mesh.numCellsX*mesh.numCellsY - row*mesh.numCellsX);
        int indexBC = (slice + 1)*mesh.numCellsX*mesh.numCellsY + (row + 1)*mesh.numCellsX + (col + 1);
        if(DC[index] < 1e-15)
        {
            DC[index] = 0;
            BC[indexBC] = 2;    // set Neumann BC with zero flux
        }
    }

    // If any phase is impermeable, need to find all participating media

    FloodFill3D_DeffSetup(&mesh, BC, DC);

    return 0;
}

#endif