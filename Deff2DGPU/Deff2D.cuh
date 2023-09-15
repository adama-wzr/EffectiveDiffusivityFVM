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

typedef struct{
  float DCsolid;					// diffusion coefficient of trace species in solid phase
  float DCfluid;					// diffusion coefficient of trace species in fluid phase
  int MeshIncreaseX;				// Mesh refinement in x-direction
  int MeshIncreaseY;				// Mesh refinement in y-direction
  float CLeft;						// Concentration of trace species in left boundary
  float CRight;					// Concentration of trace species in right boundary
  long int MAX_ITER;				// Max iterations
  float ConvergeCriteria;			// Convergence Criteria
  char* inputFilename;				// Input filename
  char* outputFilename;				// Output filename
  int printCmap;					// print concentration map (true/false) flag
  char* CMapName;					// Concentration map name
  int verbose;						// verbose flag
  int BatchFlag;					// Batch flag
  int NumImg;						// Number of images in the batch
}options;


int printOptions(options* opts){
	if(opts->BatchFlag == 0){
		printf("--------------------------------------\n\n");
		printf("Current selected options:\n\n");
		printf("--------------------------------------\n");
		printf("DC Fluid = %.2f\n", opts->DCfluid);
		printf("DC Solid = %.2f\n", opts->DCsolid);
		printf("Concentration Left = %.2f\n", opts->CLeft);
		printf("Concentration Right = %.2f\n", opts->CRight);
		printf("Mesh Amp. X = %d\n", opts->MeshIncreaseX);
		printf("Mesh Amp. Y = %d\n", opts->MeshIncreaseY);
		printf("Maximum Iterations = %ld\n", opts->MAX_ITER);
		printf("Convergence = %.10f\n", opts->ConvergeCriteria);
		printf("Name of input image: %s\n", opts->inputFilename);
		printf("Name of output file: %s\n", opts->outputFilename);

		if(opts->printCmap == 0){
			printf("Print Temperature Map = False\n");
		} else{
			printf("Temperature Map Name = %s\n", opts->CMapName);
		}
		printf("--------------------------------------\n\n");
	} else if(opts->BatchFlag == 1){
		printf("--------------------------------------\n\n");
		printf("Running Image Batch:\n\n");
		printf("DC Fluid = %.2f\n", opts->DCfluid);
		printf("DC Solid = %.2f\n", opts->DCsolid);
		printf("Concentration Left = %.2f\n", opts->CLeft);
		printf("Concentration Right = %.2f\n", opts->CRight);
		printf("Mesh Amp. X = %d\n", opts->MeshIncreaseX);
		printf("Mesh Amp. Y = %d\n", opts->MeshIncreaseY);
		printf("Maximum Iterations = %ld\n", opts->MAX_ITER);
		printf("Convergence = %.10f\n", opts->ConvergeCriteria);
		printf("Name of output file: %s\n", opts->outputFilename);
		printf("Number of files to run: %d\n", opts->NumImg);
		if (opts->printCmap == 1){
			printf("Printing Temperature Distribution for all images.\n");
		} else{
			printf("No temperature maps will be printed.\n");
		}
		printf("--------------------------------------\n\n");
	} else{
		printf("Options entered are not valid, code will exit.\n");
		return 1;
	}	

	return 0;
}



int readInputFile(char* FileName, options* opts){

	/*
		readInputFile Function:
		Inputs:
			- FileName: pointer to where the input file name is stored.
			- struct options: pass a struct with the options.
		Outputs: None

		Function reads the input file and stores the options in the opts struct.
	*/

	std::string myText;

	char tempC[1000];
	float tempD;
	char tempFilenames[1000];
	std::ifstream InputFile(FileName);

	// initialize the pointers so they are not random

	opts->inputFilename=(char*)malloc(1000*sizeof(char));
	opts->outputFilename=(char*)malloc(1000*sizeof(char));
	opts->CMapName=(char*)malloc(1000*sizeof(char));
	while(std::getline(InputFile, myText)){

	 	sscanf(myText.c_str(), "%s %f", tempC, &tempD);
	 	if (strcmp(tempC, "Ds:") == 0){
	 		opts->DCsolid = tempD;
	 	}else if(strcmp(tempC, "Df:") == 0){
	 		opts->DCfluid = tempD;

	 	}else if(strcmp(tempC, "MeshAmpX:") == 0){
	 		opts->MeshIncreaseX = (int)tempD;

	 	}else if(strcmp(tempC, "MeshAmpY:") == 0){
	 		opts->MeshIncreaseY = (int)tempD;

	 	}else if(strcmp(tempC, "InputName:") == 0){
	 		sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
	 		strcpy(opts->inputFilename, tempFilenames);

	 	}else if(strcmp(tempC, "CR:") == 0){
	 		opts->CRight = tempD;

	 	}else if(strcmp(tempC, "CL:") == 0){
	 		opts->CLeft = tempD;

	 	}else if(strcmp(tempC, "OutputName:") == 0){
	 		sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
	 		strcpy(opts->outputFilename, tempFilenames);

	 	}else if(strcmp(tempC, "printCMap:") == 0){
	 		opts->printCmap = (int)tempD;

	 	}else if(strcmp(tempC, "CMapName:") == 0){
	 		sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
	 		strcpy(opts->CMapName, tempFilenames);

	 	}else if(strcmp(tempC, "Convergence:") == 0){
	 		opts->ConvergeCriteria = tempD;

	 	}else if(strcmp(tempC, "MaxIter:") == 0){
	 		opts->MAX_ITER = (long int)tempD;

	 	}else if(strcmp(tempC, "Verbose:") == 0){
	 		opts->verbose = (int)tempD;

	 	} else if(strcmp(tempC, "RunBatch:") == 0){
	 		opts->BatchFlag = (int)tempD;
	 	} else if(strcmp(tempC, "NumImages:") == 0){
	 		opts->NumImg = (int)tempD;
	 	}
	}
	
	InputFile.close();



	if(opts->verbose == 1){
		printOptions(opts);
	} else if(opts->verbose != 0){
		printf("Please enter a value of 0 or 1 for 'verbose'. Default = 0.\n");
	}
	return 0;
}