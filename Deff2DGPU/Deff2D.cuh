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
	double DCsolid;			 // diffusion coefficient of trace species in solid phase
	double DCfluid;			 // diffusion coefficient of trace species in fluid phase
	double DCgas;			 // diffusion coefficient of trace species in gas phase
	int MeshIncreaseX;		 // Mesh refinement in x-direction
	int MeshIncreaseY;		 // Mesh refinement in y-direction
	double CLeft;			 // Concentration of trace species in left boundary
	double CRight;			 // Concentration of trace species in right boundary
	long int MAX_ITER;		 // Max iterations
	double ConvergeCriteria; // Convergence Criteria
	char *inputFilename;	 // Input filename
	char *outputFilename;	 // Output filename
	int printCmap;			 // print concentration map (true/false) flag
	char *CMapName;			 // Concentration map name
	int verbose;			 // verbose flag
	int BatchFlag;			 // Batch flag
	int NumImg;				 // Number of images in the batch
	int nPhase;
} options;

typedef struct
{
	int Width;
	int Height;
	int nChannels;
	double porosity;
	double SVF;
	double LVF;
	double gpuTime;
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

// define pair for coords a* algorithms

typedef std::pair<int, int> coordPair;

// GPU Jacobi-Iteration Standard Over-Relaxation (SOR) Kernel

__global__ void updateX_SOR(double* A, double* x, double* b, double* xNew, meshInfo mesh)
{
	unsigned int myRow = blockIdx.x * blockDim.x + threadIdx.x;
	double w = 2.0/3.0;

	if (myRow < mesh.nElements){
		double sigma = 0;
		for(int j = 1; j<5; j++){
			if(A[myRow*5 + j] !=0){
				if(j == 1){
					sigma += A[myRow*5 + j]*x[myRow - 1];
				} else if(j == 2){
					sigma += A[myRow*5 + j]*x[myRow + 1];
				} else if(j == 3){
					sigma += A[myRow*5 + j]*x[myRow + mesh.numCellsX];
				} else if(j == 4){
					sigma += A[myRow*5 + j]*x[myRow - mesh.numCellsX];
				}
			}
		}
		xNew[myRow] = (1.0-w)*x[myRow] +  w/A[myRow*5 + 0] * (b[myRow] - sigma);
	}
		
}

// Jacobi-Iteration GPU Kernel

__global__ void updateX_V1(double* A, double* x, double* b, double* xNew, meshInfo mesh)
{
	unsigned int myRow = blockIdx.x * blockDim.x + threadIdx.x;

	if (myRow < mesh.nElements){
		double sigma = 0;
		for(int j = 1; j<5; j++){
			if(A[myRow*5 + j] !=0){
				if(j == 1){
					sigma += A[myRow*5 + j]*x[myRow - 1];
				} else if(j == 2){
					sigma += A[myRow*5 + j]*x[myRow + 1];
				} else if(j == 3){
					sigma += A[myRow*5 + j]*x[myRow + mesh.numCellsX];
				} else if(j == 4){
					sigma += A[myRow*5 + j]*x[myRow - mesh.numCellsX];
				}
			}
		}
		xNew[myRow] = 1/A[myRow*5 + 0] * (b[myRow] - sigma);
	}
		
}


int printOptions(options* opts){
	/*
		Function prints the selected user options if verbose = true
	*/
	if(opts->BatchFlag == 0){
		printf("--------------------------------------\n\n");
		printf("Current selected options:\n\n");
		printf("--------------------------------------\n");
		printf("Number of Phases = %d\n", opts->nPhase);
		printf("DC Fluid = %1.3e\n", opts->DCfluid);
		printf("DC Solid = %1.3e\n", opts->DCsolid);
		printf("DC Gas = %1.3e\n", opts->DCgas);
		printf("Concentration Left = %.2f\n", opts->CLeft);
		printf("Concentration Right = %.2f\n", opts->CRight);
		printf("Mesh Amp. X = %d\n", opts->MeshIncreaseX);
		printf("Mesh Amp. Y = %d\n", opts->MeshIncreaseY);
		printf("Maximum Iterations = %ld\n", opts->MAX_ITER);
		printf("Convergence = %.10f\n", opts->ConvergeCriteria);
		printf("Name of input image: %s\n", opts->inputFilename);
		printf("Name of output file: %s\n", opts->outputFilename);

		if(opts->printCmap == 0){
			printf("Print Concentration Map = False\n");
		} else{
			printf("Concentration Map Name = %s\n", opts->CMapName);
		}
		printf("--------------------------------------\n\n");
	} else if(opts->BatchFlag == 1){
		printf("--------------------------------------\n\n");
		printf("Running Image Batch:\n\n");
		printf("Number of Phases = %d\n", opts->nPhase);
		printf("DC Fluid = %1.3e\n", opts->DCfluid);
		printf("DC Solid = %1.3e\n", opts->DCsolid);
		printf("DC Gas = %1.3e\n", opts->DCgas);
		printf("Concentration Left = %.2f\n", opts->CLeft);
		printf("Concentration Right = %.2f\n", opts->CRight);
		printf("Mesh Amp. X = %d\n", opts->MeshIncreaseX);
		printf("Mesh Amp. Y = %d\n", opts->MeshIncreaseY);
		printf("Maximum Iterations = %ld\n", opts->MAX_ITER);
		printf("Convergence = %.10f\n", opts->ConvergeCriteria);
		printf("Name of output file: %s\n", opts->outputFilename);
		printf("Number of files to run: %d\n", opts->NumImg);
		if (opts->printCmap == 1){
			printf("Printing Concentration Distribution for all images.\n");
		} else{
			printf("No Concentration maps will be printed.\n");
		}
		printf("--------------------------------------\n\n");
	} else{
		printf("Options entered are not valid, code will exit.\n");
		return 1;
	}	

	return 0;
}

int outputSingle(options opts, meshInfo mesh, simulationInfo myImg)
{
	FILE *OUTPUT;
	// imgNum, porosity,PathFlag,Deff,Time,nElements,converge,ds,df

	OUTPUT = fopen(opts.outputFilename, "a+");
	fprintf(OUTPUT, "imgNum,porosity,PathFlag,Deff,Time,nElements,converge,ds,df\n");
	fprintf(OUTPUT, "%s,%f,%d,%f,%f,%d,%f,%f,%f\n", opts.inputFilename, myImg.porosity, myImg.PathFlag, myImg.deff, myImg.gpuTime / 1000, mesh.nElements, myImg.conv,
			opts.DCsolid, opts.DCfluid);
	fclose(OUTPUT);
	return 0;
}


int outputSingle3Phase(options opts, meshInfo mesh, simulationInfo myImg)
{
	FILE *OUTPUT;
	// imgNum, porosity,PathFlag,Deff,Time,nElements,converge,ds,df

	OUTPUT = fopen(opts.outputFilename, "a+");
	// fprintf(OUTPUT, "imgNum,SVF,LVF,PathFlag,Deff,Time,nElements,converge,ds,df,dg\n");
	fprintf(OUTPUT, "%s,%f,%f,%d,%1.3e,%f,%d,%1.3e,%1.3e,%1.3e,%1.3e\n", opts.inputFilename, myImg.SVF, myImg.LVF, myImg.PathFlag, myImg.deff, myImg.gpuTime / 1000, mesh.nElements, myImg.conv,
			opts.DCsolid, opts.DCfluid, opts.DCgas);
	fclose(OUTPUT);
	return 0;
}

int outputBatch(options opts, double *output)
{
	FILE *OUTPUT;

	OUTPUT = fopen(opts.outputFilename, "a+");
	fprintf(OUTPUT, "imgNum,porosity,PathFlag,Deff,Time,nElements,converge,ds,df\n");
	for (int i = 0; i < opts.NumImg; i++)
	{
		fprintf(OUTPUT, "%d,%f,%d,%f,%f,%d,%f,%f,%f\n", i, output[i * 9 + 1], (int)output[i * 9 + 2], output[i * 9 + 3], output[i * 9 + 4], (int)output[i * 9 + 5], output[i * 9 + 6],
				output[i * 9 + 7], output[i * 9 + 8]);
	}
	fclose(OUTPUT);
	return 0;
}

int outputBatch3Phase(options opts, double *output)
{
	FILE *OUTPUT;

	OUTPUT = fopen(opts.outputFilename, "a+");
	fprintf(OUTPUT, "imgNum,SVF,LVF,PathFlag,Deff,Time,nElements,converge,ds,df,dg\n");
	for (int i = 0; i < opts.NumImg; i++)
	{
		fprintf(OUTPUT, "%d,%f,%f,%d,%1.3e,%f,%d,%1.3e,%1.3e,%1.3e,%1.3e\n", i, output[i * 10 + 1], output[i * 10 + 2], (int)output[i * 10 + 3], output[i * 10 + 4],
				output[i * 10 + 5], (int)output[i * 10 + 6], output[i * 10 + 7], output[i * 10 + 8], output[i * 10 + 9], opts.DCgas);
	}
	fclose(OUTPUT);
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
	double tempD;
	char tempFilenames[1000];
	std::ifstream InputFile(FileName);

	// initialize the pointers so they are not random

	opts->inputFilename=(char*)malloc(1000*sizeof(char));
	opts->outputFilename=(char*)malloc(1000*sizeof(char));
	opts->CMapName=(char*)malloc(1000*sizeof(char));
	while(std::getline(InputFile, myText)){

	 	sscanf(myText.c_str(), "%s %lf", tempC, &tempD);
	 	if (strcmp(tempC, "Ds:") == 0){
	 		opts->DCsolid = tempD;
	 	}else if(strcmp(tempC, "Df:") == 0){
	 		opts->DCfluid = tempD;

		}else if(strcmp(tempC, "Dg:") == 0){
	 		opts->DCgas = tempD;

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
	 	} else if(strcmp(tempC, "Phases:") == 0){
	 		opts->nPhase = (int)tempD;
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


int readImage(options opts, simulationInfo* myImg){
	/*
		readImage Function:
		Inputs:
			- imageAddress: unsigned char reference to the pointer in which the image will be read to.
			- Width: pointer to variable to store image width
			- Height: pointer to variable to store image height
			- NumofChannels: pointer to variable to store number of channels in the image.
					-> NumofChannels has to be 1, otherwise code is terminated. Please enter grayscale
						images with NumofChannels = 1.
		Outputs: None

		Function reads the image into the pointer to the array to store it.
	*/

	myImg->target_data = stbi_load(opts.inputFilename, &myImg->Width, &myImg->Height, &myImg->nChannels, 1);

	return 0;
}

double WeightedHarmonicMean(double w1, double w2, double x1, double x2){
	/*
		WeightedHarmonicMean Function:
		Inputs:
			-w1: weight of the first number
			-w2: weight of the second number
			-x1: first number to be averaged
			-x2: second number to be averaged
		Output:
			- returns H, the weighted harmonic mean between x1 and x2, using weights w1 and w2.
	*/
	double H = (w1 + w2)/(w1/x1 + w2/x2);
	return H;
}

int readImageBatch(options opts, simulationInfo* myImg, char* filename){
	/*
		readImage Function:
		Inputs:
			- imageAddress: unsigned char reference to the pointer in which the image will be read to.
			- Width: pointer to variable to store image width
			- Height: pointer to variable to store image height
			- NumofChannels: pointer to variable to store number of channels in the image.
					-> NumofChannels has to be 1, otherwise code is terminated. Please enter grayscale
						images with NumofChannels = 1.
		Outputs: None

		Function reads the image into the pointer to the array to store it.
	*/

	myImg->target_data = stbi_load(filename, &myImg->Width, &myImg->Height, &myImg->nChannels, 1);

	return 0;
}


double calcPorosity(unsigned char* imageAddress, int Width, int Height){
	/*
		calcPorosity
		Inputs:
			- imageAddress: pointer to the read image.
			- Width: original width from std_image
			- Height: original height from std_image

		Output:
			- porosity: double containing porosity.

		Function calculates porosity by counting pixels.
	*/

	double totalCells = (double)Height*Width;
	double porosity = 0;
	for(int i = 0; i<Height; i++){
		for(int j = 0; j<Width; j++){
			if(imageAddress[i*Width + j] < 150){
				porosity += 1.0/totalCells;
			}
		}
	}

	return porosity;
}


double calcFracts3D(simulationInfo* simInfo, double* D , meshInfo* mesh, options* opts){
	/*
		calcPorosity
		Inputs:
			- simInfor: pointer to struct containing simulation information
			- D: pointer to array containing local diffusion coefficients.
			- mesh: pointer to struct containing mesh information.
			- opts: pointer to user entered options struct.

		Output:
			- None

		Function calculates solid volume fraction (SVF) and liquid volume fraction (LVF).
		The results are stored in the simInfo struct.
	*/

	double totalCells = mesh->nElements;
	double SVF = 0;
	double LVF = 0;
	int Height, Width;
	Height = mesh->numCellsY;
	Width = mesh->numCellsX;

	for(int i = 0; i<Height; i++){
		for(int j = 0; j<Width; j++){
			if(D[i*Width + j] == opts->DCsolid){
				SVF += 1.0/totalCells;
			}else if(D[i*Width + j] ==  opts->DCfluid){
				LVF += 1.0/totalCells;
			}
		}
	}

	simInfo->SVF = SVF;
	simInfo->LVF = LVF;

	return 0;
}


double Residual(int numRows, int numCols, options* o, double* cmap, double* D){
	/*
		Function to calculate residual convergence (Conservation of energy in this problem)

	*/

	double dx = 1.0/numCols;
	double dy = 1.0/numRows;

	double TL = o->CLeft;
	double TR = o->CRight;

	double qE, qW, qS, qN;
	double R = 0;
	for(int row = 0; row<numRows; row++){
		for(int col = 0; col<numCols; col++){
			if(col == 0){
				qW = dy/(dx/2)*D[row*numCols + col]*(cmap[row*numCols + col] - TL);
				qE = dy/(dx)*WeightedHarmonicMean(dx/2, dx/2, D[row*numCols + col],D[row*numCols + col+1])*(cmap[row*numCols + col + 1] - cmap[row*numCols + col]);
			} else if(col == numCols - 1){
				qW = dy/(dx)*WeightedHarmonicMean(dx/2, dx/2, D[row*numCols + col],D[row*numCols + col-1])*(cmap[row*numCols + col] - cmap[row*numCols + col-1]);
				qE = dy/(dx/2)*D[row*numCols + col]*(TR - cmap[row*numCols + col]);
			} else{
				qW = dy/(dx)*WeightedHarmonicMean(dx/2, dx/2, D[row*numCols + col],D[row*numCols + col-1])*(cmap[row*numCols + col] - cmap[row*numCols + col-1]);
				qE = dy/(dx)*WeightedHarmonicMean(dx/2, dx/2, D[row*numCols + col],D[row*numCols + col+1])*(cmap[row*numCols + col + 1] - cmap[row*numCols + col]);
			}
			if(row == 0){
				qN = 0;
				qS = dy/dx*WeightedHarmonicMean(dx/2, dx/2, D[(row+1)*numCols + col], D[(row)*numCols + col])*(cmap[(row+1)*numCols + col] - cmap[row*numCols + col]);
			} else if(row == numRows - 1){
				qS = 0;
				qN = dy/dx*WeightedHarmonicMean(dx/2, dx/2, D[(row-1)*numCols + col], D[(row)*numCols + col])*(cmap[(row)*numCols + col] - cmap[(row-1)*numCols + col]);
			} else{
				qS = dy/dx*WeightedHarmonicMean(dx/2, dx/2, D[(row+1)*numCols + col], D[(row)*numCols + col])*(cmap[(row+1)*numCols + col] - cmap[row*numCols + col]);
				qN = dy/dx*WeightedHarmonicMean(dx/2, dx/2, D[(row-1)*numCols + col], D[(row)*numCols + col])*(cmap[(row)*numCols + col] - cmap[(row-1)*numCols + col]);
			}
			R += fabs(qW - qE + qN - qS);
		}
	}

	R = R/(numCols*numRows);

	return R;
}


void createCMAP(double *CMap, options *opts, meshInfo *mesh)
{
	/*
		createCMAP function:

		Inputs:
			- pointer to CMap: pointer to array containing the concentration at each grid point.
			- pointer to opts: pointer to options struct containing simulation options.
			- pointer to mesh: pointer to mesh struct containing mesh information.
		Outputs:
			- None.

		Funtion will create a .csv of the concentration distribution in the domain, using the user entered
		name for the .csv file.
	
	*/

	FILE *OUTPUT;

	OUTPUT = fopen(opts->CMapName, "w+");
	fprintf(OUTPUT, "X,Y,C\n");
	for(int i = 0; i<mesh->numCellsY; i++){
		for(int j = 0; j<mesh->numCellsX; j++){
			fprintf(OUTPUT,"%d,%d,%1.3e\n", j, i, CMap[i*mesh->numCellsX + j]);
		}
	}
	fclose(OUTPUT);
}


void createCMAPBatch(double *CMap, char* filename, meshInfo *mesh)
{
	/*
		createCMAP function:

		Inputs:
			- pointer to CMap: pointer to array containing the concentration at each grid point.
			- pointer to filename: pointer to a string containing the file name.
			- pointer to mesh: pointer to mesh struct containing mesh information.
		Outputs:
			- None.

		Funtion will create a .csv of the concentration distribution in the domain, using the user entered
		name for the .csv file.
	
	*/

	FILE *OUTPUT;

	OUTPUT = fopen(filename, "w+");
	fprintf(OUTPUT, "X,Y,C\n");
	for(int i = 0; i<mesh->numCellsY; i++){
		for(int j = 0; j<mesh->numCellsX; j++){
			fprintf(OUTPUT,"%d,%d,%1.3e\n", j, i, CMap[i*mesh->numCellsX + j]);
		}
	}
	fclose(OUTPUT);
}


int FloodFill(unsigned int* Grid, meshInfo* simInfo, simulationInfo* info)
{
	/*
		Flood Fill algorithm:

		Inputs:
			- pointer to Grid: array containing the location of solids and fluids
			- simulationInfo* simInfo: pointer to struct containing the simInfo

		Outputs:
			- None

		Function will modify the Grid array. If fluid is not participating, it will
		be assigned a flag of 2.
	*/

	int* Domain = (int *)malloc(sizeof(int)*simInfo->nElements);
	int index;

	// Step 1: Initialize all solids in the domain

	for(int row = 0; row<simInfo->numCellsY; row++){
		for(int col = 0; col<simInfo->numCellsX; col++){
			index = row*simInfo->numCellsX + col;
			if(Grid[index] == 1){
				Domain[index] = 1;
			} else{
				Domain[index] = -1;
			}
		}
	}

	// Step 2: Find fluid in the left boundary, set flag to 0, add to open list

	std::set<coordPair> cList;

	for(int row = 0; row<simInfo->numCellsY; row++){
		int indexL = row*simInfo->numCellsX;

		if(Domain[indexL] == -1){
			Domain[indexL] = 0;
			cList.insert(std::make_pair(row, 0));
		}
	}

	while(!cList.empty())
	{
		// Pop first item in the list
		coordPair pop = *cList.begin();

		// remove from open list
		cList.erase(cList.begin());

		// Get coordinates from popped item
		int row = pop.first; // first argument of the second pair
		int col = pop.second; // second argument of second pair

		if(col == simInfo->numCellsX - 1){
			info->PathFlag = 1;
		}

		/*
			Now we need to check North, South, East, and West for more fluid:
				North: row - 1, col
				South: row + 1, col
				West: row, col - 1
				East: row, col + 1
			Details:
				- No diagonals are checked.
				- Periodic BC North and South

		*/

		int tempRow, tempCol;

		// North
		tempCol = col;

		// check periodic boundary
		if(row == 0){
			tempRow = simInfo->numCellsY - 1;
		} else{
			tempRow = row - 1;
		}

		// Update list if necessary

		if(Domain[tempRow*simInfo->numCellsX + tempCol] == -1){
			Domain[tempRow*simInfo->numCellsX + tempCol] = 0;
			cList.insert(std::make_pair(tempRow, tempCol));
		}

		// South

		tempCol = col;

		// check periodic boundary

		if(row == simInfo->numCellsY - 1){
			tempRow = 0;
		} else{
			tempRow = row + 1;
		}

		// Update list if necessary

		if(Domain[tempRow*simInfo->numCellsX + tempCol] == -1){
			Domain[tempRow*simInfo->numCellsX + tempCol] = 0;
			cList.insert(std::make_pair(tempRow, tempCol));
		}

		// West

		if(col != 0){
			tempCol = col - 1;
			tempRow = row;

			if(Domain[tempRow*simInfo->numCellsX + tempCol] == -1){
				Domain[tempRow*simInfo->numCellsX + tempCol] = 0;
				cList.insert(std::make_pair(tempRow, tempCol));
			}
		}

		// East

		if(col != simInfo->numCellsX - 1){
			tempCol = col + 1;
			tempRow = row;

			if(Domain[tempRow*simInfo->numCellsX + tempCol] == -1){
				Domain[tempRow*simInfo->numCellsX + tempCol] = 0;
				cList.insert(std::make_pair(tempRow, tempCol));
			}
		}
		// repeat everything if list is still not empty
	}

	// Every flag that is still -1 means its non-participating fluid

	for(int row = 0; row<simInfo->numCellsY; row++){
		for(int col = 0; col<simInfo->numCellsX; col++){
			index = row*simInfo->numCellsX + col;
			if(Domain[index] == -1){
				Grid[index] = 2;
			}
		}
	}

	free(Domain);

	return 0;
}

int DiscretizeMatrix2D_ImpSolid(double* D, double* A, double* b, meshInfo mesh, options opts, unsigned int* Grid){
	/*
		DiscretizeMatrix2D_ImpSolid

		Inputs:
			- pointer to double array D, where local diffusion coefficients are stored
			- pointer to empty coefficient matrix
			- pointer to RHS of the system of equations
			- datastructure containing mesh information
			- datastructure with the user-entered options
			- pointer to grid. Grid contains domain phase information
		Output:
			- None

			Function creates the CoeffMatrix and RHS of the system of equations and stores them
			on the appropriate memory spaces. The difference is now we correct for impermeable solid
			as well as non-participating media.
	*/

	int index;
	double dxw, dxe, dys, dyn;
	double kw, ke, ks, kn;

	double dx, dy;
	dx = mesh.dx;
	dy = mesh.dy;

	for(int i = 0; i<mesh.numCellsY; i++){
		for(int j = 0; j<mesh.numCellsX; j++){
			// initialize everything to zeroes
			index = (i*mesh.numCellsX + j); 
			b[index] = 0;
			for(int k = 0; k<5; k++){
				A[index*5 + k] = 0;
			}
			if(Grid[index] == 1 || Grid[index] == 2){	// Correction for non-participating media
				A[index*5 + 0] = 1;
				b[index] = 0;
			} else{
				// left boundary, only P and E
				if (j == 0){
					dxe = dx;
					ke = WeightedHarmonicMean(dxe/2,dxe/2, D[index], D[index+1]);
					dxw = dx/2;
					kw = D[index];
					A[index*5 + 2] = -ke*dy/dxe;
					A[index*5 + 0] += (ke*dy/dxe + kw*dy/dxw);
					b[index] += opts.CLeft*kw*dy/dxw;
				} else if(j == mesh.numCellsX - 1){		// Right boundary, only P and W
					dxw = dx;
					kw = WeightedHarmonicMean(dxw/2,dxw/2, D[index], D[index-1]);
					dxe = dx/2;
					ke = D[index];
					A[index*5 + 1] = -kw*dy/dxw;
					A[index*5 + 0] += (ke*dy/dxe + kw*dy/dxw);
					b[index] += opts.CRight*ke*dy/dxe;
				} else{								// P, W, and E
					dxw = dx;
					kw = WeightedHarmonicMean(dxw/2,dxw/2, D[index], D[index-1]);
					dxe = dx;
					ke = WeightedHarmonicMean(dxe/2,dxe/2, D[index], D[index+1]);
					A[index*5 + 1] = -kw*dy/dxw;
					A[index*5 + 2] = -ke*dy/dxe;
					A[index*5 + 0] += (ke*dy/dxe + kw*dy/dxw);
				}
				// top boundary, only S and P
				if (i == 0){
					dyn = dy/2;
					kn = D[index];
					dys = dy;
					ks = WeightedHarmonicMean(dys/2, dys/2, D[index + mesh.numCellsX], D[index]);
					A[index*5 + 3] = -ks*dx/dys;
					A[index*5 + 0] += (ks*dx/dys);
				}else if(i == mesh.numCellsY - 1){
					dyn = dy;
					kn = WeightedHarmonicMean(dyn/2, dyn/2, D[index], D[index - mesh.numCellsX]);
					dys = dy/2;
					ks = D[index];
					A[index*5 + 4] = -kn*dx/dyn;
					A[index*5 + 0] += kn*dx/dyn;
				} else{
					dyn = dy;
					kn = WeightedHarmonicMean(dyn/2, dyn/2, D[index], D[index - mesh.numCellsX]);
					dys = dy;
					ks = WeightedHarmonicMean(dys/2, dys/2, D[index + mesh.numCellsX], D[index]);
					A[index*5 + 3] = -ks*dx/dys;
					A[index*5 + 4] = -kn*dx/dyn;
					A[index*5 + 0] += (kn*dx/dyn + ks*dx/dys);
				}
			}
		}
	}

	return 0;



}


int DiscretizeMatrix2D(double* D, double* A, double* b, meshInfo mesh, options opts){
	/*
		DiscretizeMatrix2D

		Inputs:
			- pointer to double array D, where local diffusion coefficients are stored
			- pointer to empty coefficient matrix
			- pointer to RHS of the system of equations
			- datastructure containing mesh information
			- datastructure with the user-entered options
		Output:
			- None

			Function creates the CoeffMatrix and RHS of the system of equations and stores them
			on the appropriate memory spaces.
	*/

	int index;
	double dxw, dxe, dys, dyn;
	double kw, ke, ks, kn;

	double dx, dy;
	dx = mesh.dx;
	dy = mesh.dy;

	for(int i = 0; i<mesh.numCellsY; i++){
		for(int j = 0; j<mesh.numCellsX; j++){
			// initialize everything to zeroes
			index = (i*mesh.numCellsX + j); 
			b[index] = 0;
			for(int k = 0; k<5; k++){
				A[index*5 + k] = 0;
			}
			// left boundary, only P and E
			if (j == 0){
				dxe = dx;
				ke = WeightedHarmonicMean(dxe/2,dxe/2, D[index], D[index+1]);
				dxw = dx/2;
				kw = D[index];
				A[index*5 + 2] = -ke*dy/dxe;
				A[index*5 + 0] += (ke*dy/dxe + kw*dy/dxw);
				b[index] += opts.CLeft*kw*dy/dxw;
			} else if(j == mesh.numCellsX - 1){		// Right boundary, only P and W
				dxw = dx;
				kw = WeightedHarmonicMean(dxw/2,dxw/2, D[index], D[index-1]);
				dxe = dx/2;
				ke = D[index];
				A[index*5 + 1] = -kw*dy/dxw;
				A[index*5 + 0] += (ke*dy/dxe + kw*dy/dxw);
				b[index] += opts.CRight*ke*dy/dxe;
			} else{								// P, W, and E
				dxw = dx;
				kw = WeightedHarmonicMean(dxw/2,dxw/2, D[index], D[index-1]);
				dxe = dx;
				ke = WeightedHarmonicMean(dxe/2,dxe/2, D[index], D[index+1]);
				A[index*5 + 1] = -kw*dy/dxw;
				A[index*5 + 2] = -ke*dy/dxe;
				A[index*5 + 0] += (ke*dy/dxe + kw*dy/dxw);
			}
			// top boundary, only S and P
			if (i == 0){
				dyn = dy/2;
				kn = D[index];
				dys = dy;
				ks = WeightedHarmonicMean(dys/2, dys/2, D[index + mesh.numCellsX], D[index]);
				A[index*5 + 3] = -ks*dx/dys;
				A[index*5 + 0] += (ks*dx/dys);
			}else if(i == mesh.numCellsY - 1){
				dyn = dy;
				kn = WeightedHarmonicMean(dyn/2, dyn/2, D[index], D[index - mesh.numCellsX]);
				dys = dy/2;
				ks = D[index];
				A[index*5 + 4] = -kn*dx/dyn;
				A[index*5 + 0] += kn*dx/dyn;
			} else{
				dyn = dy;
				kn = WeightedHarmonicMean(dyn/2, dyn/2, D[index], D[index - mesh.numCellsX]);
				dys = dy;
				ks = WeightedHarmonicMean(dys/2, dys/2, D[index + mesh.numCellsX], D[index]);
				A[index*5 + 3] = -ks*dx/dys;
				A[index*5 + 4] = -kn*dx/dyn;
				A[index*5 + 0] += (kn*dx/dyn + ks*dx/dys);
			}
		}
	}

	return 0;
}

int initializeGPU(double **d_x_vec, double **d_temp_x_vec, double **d_RHS, double **d_Coeff, meshInfo mesh){

	// Set device, when cudaStatus is called give status of assigned device.
	// This is important to know if we are running out of GPU space
	cudaError_t cudaStatus = cudaSetDevice(0);

	// Start by allocating space in GPU memory

	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		getchar();
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&(*d_x_vec), mesh.nElements*sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		getchar();
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&(*d_temp_x_vec), mesh.nElements*sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		getchar();
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&(*d_RHS), mesh.nElements*sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		getchar();
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&(*d_Coeff), mesh.nElements*sizeof(double)*5);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		getchar();
        return 0;
    }

    // Set GPU buffers (initializing matrices to 0)

     // Memset GPU buffers
    cudaStatus = cudaMemset((*d_x_vec),0, mesh.nElements*sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
		getchar();
        return 0;
    }

	// Memset GPU buffers
    cudaStatus = cudaMemset((*d_temp_x_vec),0, mesh.nElements*sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
		getchar();
        return 0;
    }

     // Memset GPU buffers
    cudaStatus = cudaMemset((*d_RHS),0, mesh.nElements*sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
		getchar();
        return 0;
    }

	// Memset GPU buffers
    cudaStatus = cudaMemset((*d_Coeff),0, 5*mesh.nElements*sizeof(double));		// coefficient matrix has the 5 main diagonals for all elements
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
		getchar();
        return 0;
    }

    return 1;
}

void unInitializeGPU(double **d_x_vec, double **d_temp_x_vec, double **d_RHS, double **d_Coeff)
{
	cudaError_t cudaStatus;

	if((*d_x_vec)!=NULL)
    cudaStatus = cudaFree((*d_x_vec));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed!");
        return;
    }

	if((*d_temp_x_vec)!=NULL)
    cudaStatus = cudaFree((*d_temp_x_vec));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed!");
        return;
    }

	if((*d_Coeff)!=NULL)
    cudaStatus = cudaFree((*d_Coeff));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed!");
        return;
    }

	if((*d_RHS)!=NULL)
    cudaStatus = cudaFree((*d_RHS));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed!");
        return;
    }    

	cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		getchar();
        return;
    }
}


int JacobiGPUPreCond(double *arr, double *sol, double *x_vec, double *temp_x_vec, options opts,
			  double *d_x_vec, double *d_temp_x_vec, double *d_Coeff, double *d_RHS, double *MFL, double *MFR, double *D, meshInfo mesh, simulationInfo *myImg)
{

	int iterCount = 0;
	double Res = 1;
	int threads_per_block = 160;
	int numBlocks = mesh.nElements / threads_per_block + 1;
	double deffNew = 1;
	double deffOld = 5;
	double percentChange = 100.0;
	int iterToCheck = 10000;
	double Q1, Q2;
	double qAvg = 0;
	double dx, dy;
	int numRows = mesh.numCellsY;
	int numCols = mesh.numCellsX;
	const char *str = (char *)malloc(1024); // To store error string

	dx = mesh.dx;
	dy = mesh.dy;

	int nRows = mesh.nElements; // number of rows in the coefficient matrix
	int nCols = 5;				// number of cols in the coefficient matrix

	// Initialize temp_x_vec

	for (int i = 0; i < nRows; i++)
	{
		temp_x_vec[i] = x_vec[i];
	}

	// Copy arrays into GPU memory

	cudaError_t cudaStatus = cudaMemcpy(d_temp_x_vec, temp_x_vec, sizeof(double) * nRows, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "temp_x_vec cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}
	cudaStatus = cudaMemcpy(d_RHS, sol, sizeof(double) * nRows, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_RHS cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}
	cudaStatus = cudaMemcpy(d_Coeff, arr, sizeof(double) * nRows * nCols, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_Coeff cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}

	// Declare event to get time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	while (iterCount < opts.MAX_ITER && opts.ConvergeCriteria < fabs(percentChange))
	{
		// Call Kernel to Calculate new x-vector

		// updateX_V1<<<numBlocks, threads_per_block>>>(d_Coeff, d_temp_x_vec, d_RHS, d_x_vec, mesh);
		updateX_SOR<<<numBlocks, threads_per_block>>>(d_Coeff, d_temp_x_vec, d_RHS, d_x_vec, mesh);

		cudaDeviceSynchronize();

		// Convergence related material

		if (iterCount % iterToCheck == 0)
		{
			cudaStatus = cudaMemcpy(x_vec, d_x_vec, sizeof(double) * nRows, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "x_vec cudaMemcpy failed!");
				str = cudaGetErrorString(cudaStatus);
				fprintf(stderr, "CUDA Error!:: %s\n", str);
			}
			Q1 = 0;
			Q2 = 0;
			for (int j = 0; j < numRows; j++)
			{
				MFL[j] = D[j * numCols] * (x_vec[j * numCols] - opts.CLeft) / (dx / 2.0);
				MFR[j] = D[(j + 1) * numCols - 1] * (opts.CRight - x_vec[(j + 1) * numCols - 1]) / (dx / 2.0);
				Q1 += MFL[j];
				Q2 += MFR[j];
			}
			Q1 = Q1;
			Q2 = Q2;
			qAvg = (Q1 + Q2) / (2.0 * numRows*dx/dy);
			deffNew = qAvg / ((opts.CRight - opts.CLeft));
			percentChange = (deffOld - deffNew)/(deffOld);
			// Res = Residual(numRows, numCols, &opts, x_vec, D);
			if (opts.verbose == 1 && opts.BatchFlag == 0)
			{
				// printf("Iteration = %d, Deff = %1.3e, Residual = %1.3e, Deff Change = %2.3f\n", iterCount, deffNew / opts.DCfluid, Res);
				printf("Iteration = %d, Deff = %1.3e, Deff Change = %1.3e\n", iterCount, deffNew / opts.DCfluid, percentChange);
			}
			deffOld = deffNew;
		}

		// update x vector
		// d_temp_x_vec = d_x_vec;

		cudaStatus = cudaMemcpy(d_temp_x_vec, d_x_vec, sizeof(double) * nRows, cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_temp_x_vec cudaMemcpy failed!");
			str = cudaGetErrorString(cudaStatus);
			fprintf(stderr, "CUDA Error!:: %s\n", str);
		}

		// Update iteration count
		iterCount++;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaStatus = cudaMemcpy(x_vec, d_x_vec, sizeof(double) * nRows, cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "x_vec cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}

	return iterCount;
}


int JacobiGPU(double *arr, double *sol, double *x_vec, double *temp_x_vec, options opts,
			  double *d_x_vec, double *d_temp_x_vec, double *d_Coeff, double *d_RHS, double *MFL, double *MFR, double *D, meshInfo mesh, simulationInfo *myImg)
{

	int iterCount = 0;
	double Res = 1;
	int threads_per_block = 160;
	int numBlocks = mesh.nElements / threads_per_block + 1;
	double deffNew = 1;
	double deffOld = 5;
	double percentChange = 100.0;
	int iterToCheck = 10000;
	double Q1, Q2;
	double qAvg = 0;
	double dx, dy;
	int numRows = mesh.numCellsY;
	int numCols = mesh.numCellsX;
	const char *str = (char *)malloc(1024); // To store error string

	dx = mesh.dx;
	dy = mesh.dy;

	int nRows = mesh.nElements; // number of rows in the coefficient matrix
	int nCols = 5;				// number of cols in the coefficient matrix

	// Initialize temp_x_vec

	for (int i = 0; i < nRows; i++)
	{
		temp_x_vec[i] = x_vec[i];
	}

	// FILE *OUT;

	// OUT = fopen("ConvData.csv", "w");

	// fprintf(OUT, "iter,Deff,R\n");

	// Copy arrays into GPU memory

	cudaError_t cudaStatus = cudaMemcpy(d_temp_x_vec, temp_x_vec, sizeof(double) * nRows, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "temp_x_vec cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}
	cudaStatus = cudaMemcpy(d_RHS, sol, sizeof(double) * nRows, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_RHS cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}
	cudaStatus = cudaMemcpy(d_Coeff, arr, sizeof(double) * nRows * nCols, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_Coeff cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}

	// Declare event to get time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	while (iterCount < opts.MAX_ITER && opts.ConvergeCriteria < fabs(percentChange))
	{
		// Call Kernel to Calculate new x-vector

		// updateX_V1<<<numBlocks, threads_per_block>>>(d_Coeff, d_temp_x_vec, d_RHS, d_x_vec, mesh);
		updateX_SOR<<<numBlocks, threads_per_block>>>(d_Coeff, d_temp_x_vec, d_RHS, d_x_vec, mesh);

		cudaDeviceSynchronize();

		// Convergence related material

		if (iterCount % iterToCheck == 0)
		{
			cudaStatus = cudaMemcpy(x_vec, d_x_vec, sizeof(double) * nRows, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "x_vec cudaMemcpy failed!");
				str = cudaGetErrorString(cudaStatus);
				fprintf(stderr, "CUDA Error!:: %s\n", str);
			}
			Q1 = 0;
			Q2 = 0;
			for (int j = 0; j < numRows; j++)
			{
				MFL[j] = D[j * numCols] * (x_vec[j * numCols] - opts.CLeft) / (dx / 2.0);
				MFR[j] = D[(j + 1) * numCols - 1] * (opts.CRight - x_vec[(j + 1) * numCols - 1]) / (dx / 2.0);
				Q1 += MFL[j];
				Q2 += MFR[j];
			}
			Q1 = Q1;
			Q2 = Q2;
			qAvg = (Q1 + Q2) / (2.0 * numRows*dx/dy);
			deffNew = qAvg / ((opts.CRight - opts.CLeft));
			percentChange = (deffOld - deffNew)/(deffOld);
			// Res = Residual(numRows, numCols, &opts, x_vec, D);
			if (opts.verbose == 1 && opts.BatchFlag == 0)
			{
				// printf("Iteration = %d, Deff = %1.3e, Residual = %1.3e, Deff Change = %2.3f\n", iterCount, deffNew / opts.DCfluid, Res);
				printf("Iteration = %d, Deff = %1.3e, Deff Change = %1.3e\n", iterCount, deffNew / opts.DCfluid, percentChange);
				// fprintf(OUT,"%d,%1.3e,%1.3e\n",iterCount, deffNew, percentChange);
			}
			deffOld = deffNew;
			// myImg->conv = Res;
			myImg->conv = percentChange;
		}

		// update x vector
		// d_temp_x_vec = d_x_vec;

		cudaStatus = cudaMemcpy(d_temp_x_vec, d_x_vec, sizeof(double) * nRows, cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_temp_x_vec cudaMemcpy failed!");
			str = cudaGetErrorString(cudaStatus);
			fprintf(stderr, "CUDA Error!:: %s\n", str);
		}

		// Update iteration count
		iterCount++;
	}

	// fclose(OUT);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaStatus = cudaMemcpy(x_vec, d_x_vec, sizeof(double) * nRows, cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "x_vec cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}

	myImg->deff = deffNew;

	myImg->gpuTime += elapsedTime;

	return iterCount;
}

int SingleSim3Phase(options opts)
{
	/*
		Function to read a single image and simulate the effective diffusivity for 3 phases.
		Results are stored on the output file.

		Inputs:
			Datastructure with user-defined simulation options
		Outputs:
			none
	*/

	// Define data structures

	simulationInfo myImg;

	meshInfo mesh;

	// first step is to read the image

	readImage(opts, &myImg);

	// Error messages

	if (myImg.nChannels != 1)
	{
		printf("Error: please enter a grascale image with 1 channel.\n Current number of channels = %d\n", myImg.nChannels);
		return 1;
	}

	// Define number of cells in each direction

	mesh.numCellsX = myImg.Width * opts.MeshIncreaseX;
	mesh.numCellsY = myImg.Height * opts.MeshIncreaseY;
	mesh.nElements = mesh.numCellsX * mesh.numCellsY;
	mesh.dx = 1.0 / mesh.numCellsX;
	mesh.dy = 1.0 / mesh.numCellsY;

	// Use pathfinding algorithm

	myImg.PathFlag = false;

	// Declare search boundaries for the domain

	unsigned int *Grid = (unsigned int *)malloc(sizeof(unsigned int) * mesh.numCellsX * mesh.numCellsY);

	// Only solid matters, as it is impermeable

	for (int i = 0; i < mesh.numCellsY; i++)
	{
		for (int j = 0; j < mesh.numCellsX; j++)
		{
			if (myImg.target_data[i * myImg.Width + j] > 200)
			{
				Grid[i * myImg.Width + j] = 1;
			}
			else
			{
				Grid[i * myImg.Width + j] = 0;
			}
		}
	}

	// Search path. This function will also mark the non-participating fluid

	FloodFill(Grid, &mesh, &myImg);

	// For this algorithm we continue whether there was a path or not

	// Diffusion coefficients

	double DCF = opts.DCfluid;
	double DCG = opts.DCgas;
	double DCS = opts.DCsolid;

	// Declare useful arrays
	double *D = (double *)malloc(sizeof(double) * mesh.numCellsX * mesh.numCellsY); // Grid matrix containing the diffusion coefficient of each cell with appropriate mesh
	double *MFL = (double *)malloc(sizeof(double) * mesh.numCellsY);				// mass flux in the left boundary
	double *MFR = (double *)malloc(sizeof(double) * mesh.numCellsY);				// mass flux in the right boundary

	double *CoeffMatrix = (double *)malloc(sizeof(double) * mesh.nElements * 5);		// array will be used to store our coefficient matrix
	double *RHS = (double *)malloc(sizeof(double) * mesh.nElements);					// array used to store RHS of the system of equations
	double *ConcentrationDist = (double *)malloc(sizeof(double) * mesh.nElements);		// array used to store the solution to the system of equations
	double *temp_ConcentrationDist = (double *)malloc(sizeof(double) * mesh.nElements); // array used to store the solution to the system of equations

	// Initialize the concentration map with a linear gradient between the two boundaries
	for (int i = 0; i < mesh.numCellsY; i++)
	{
		for (int j = 0; j < mesh.numCellsX; j++)
		{
			ConcentrationDist[i * mesh.numCellsX + j] = (double)j / mesh.numCellsX * (opts.CRight - opts.CLeft) + opts.CLeft;
		}
	}

	// Zero the time

	myImg.gpuTime = 0;

	// Declare GPU arrays

	double *d_x_vec = NULL;
	double *d_temp_x_vec = NULL;

	double *d_Coeff = NULL;
	double *d_RHS = NULL;

	// Initialize the GPU arrays

	if (!initializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff, mesh))
	{
		printf("\n Error when allocating space in GPU");
		unInitializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff);
		return 0;
	}

	// Populate D according to DCF, DCS, DCG, and target image. Mesh amplification is employed at this step
	// 	on converting the actual 2D image into a simulation domain.

	/*

	Target Grayscale Image Requirements:
	- Solid = 255
	- Fluid = 150
	- Gas = 0

	*/

	bool preCond = true;

	if (preCond == false)
	{
		// No preconditioning necessary, proceed normally
		for (int i = 0; i < mesh.numCellsY; i++)
		{
			MFL[i] = 0;
			MFR[i] = 0;
			for (int j = 0; j < mesh.numCellsX; j++)
			{
				int targetIndexRow = i / opts.MeshIncreaseY;
				int targetIndexCol = j / opts.MeshIncreaseX;
				if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] > 200)
				{
					D[i * mesh.numCellsX + j] = DCS;
				}
				else if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] < 50)
				{
					D[i * mesh.numCellsX + j] = DCG;
				}
				else
				{
					D[i * mesh.numCellsX + j] = DCF;
				}
			}
		}

		// Calculate phase fractions

		calcFracts3D(&myImg, D, &mesh, &opts);

		// Now that we have all pieces, generate the coefficient matrix

		DiscretizeMatrix2D_ImpSolid(D, CoeffMatrix, RHS, mesh, opts, Grid);

		// Solve with GPU
		int iter_taken = 0;
		iter_taken = JacobiGPU(CoeffMatrix, RHS, ConcentrationDist, temp_ConcentrationDist, opts,
							   d_x_vec, d_temp_x_vec, d_Coeff, d_RHS, MFL, MFR, D, mesh, &myImg);

		if (opts.verbose == 1)
		{
			printf("Iterations taken = %d\n", iter_taken);
		}
	}
	else
	{
		// Pre-condition, then solve
		double DCG_Temp = 10;
		int preCondStage = 1;

		// save original settings
		double originalTol = opts.ConvergeCriteria;
		double originalMaxIter = opts.MAX_ITER;

		// Decrease the strictness of convergence for the pre-conditioner

		opts.ConvergeCriteria = originalTol * 10;
		opts.MAX_ITER = 1e6;

		while (DCG_Temp < DCG)
		{
			if (opts.verbose == 1)
			{
				printf("Pre-Cond Stage %d: DCG = %1.3e\n", preCondStage, DCG_Temp);
			}
			for (int i = 0; i < mesh.numCellsY; i++)
			{
				MFL[i] = 0;
				MFR[i] = 0;
				for (int j = 0; j < mesh.numCellsX; j++)
				{
					int targetIndexRow = i / opts.MeshIncreaseY;
					int targetIndexCol = j / opts.MeshIncreaseX;
					if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] > 200)
					{
						D[i * mesh.numCellsX + j] = DCS;
					}
					else if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] < 50)
					{
						D[i * mesh.numCellsX + j] = DCG_Temp;
					}
					else
					{
						D[i * mesh.numCellsX + j] = DCF;
					}
				}
			}

			// Now that we have all pieces, generate the coefficient matrix

			DiscretizeMatrix2D_ImpSolid(D, CoeffMatrix, RHS, mesh, opts, Grid);

			// Solve with GPU
			int iter_taken = 0;
			iter_taken = JacobiGPUPreCond(CoeffMatrix, RHS, ConcentrationDist, temp_ConcentrationDist, opts,
										  d_x_vec, d_temp_x_vec, d_Coeff, d_RHS, MFL, MFR, D, mesh, &myImg);

			if (opts.verbose == 1)
			{
				printf("Iterations taken = %d\n", iter_taken);
			}

			DCG_Temp = DCG_Temp * 10;
			preCondStage++;
		}

		// Pre-Conditioning done, solve actual system

		opts.ConvergeCriteria = originalTol;
		opts.MAX_ITER = originalMaxIter;

		// No preconditioning necessary, proceed normally
		for (int i = 0; i < mesh.numCellsY; i++)
		{
			MFL[i] = 0;
			MFR[i] = 0;
			for (int j = 0; j < mesh.numCellsX; j++)
			{
				int targetIndexRow = i / opts.MeshIncreaseY;
				int targetIndexCol = j / opts.MeshIncreaseX;
				if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] > 200)
				{
					D[i * mesh.numCellsX + j] = DCS;
				}
				else if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] < 50)
				{
					D[i * mesh.numCellsX + j] = DCG;
				}
				else
				{
					D[i * mesh.numCellsX + j] = DCF;
				}
			}
		}

		// Calculate phase fractions

		calcFracts3D(&myImg, D, &mesh, &opts);

		// Now that we have all pieces, generate the coefficient matrix

		DiscretizeMatrix2D_ImpSolid(D, CoeffMatrix, RHS, mesh, opts, Grid);

		// Solve with GPU
		int iter_taken = 0;
		iter_taken = JacobiGPU(CoeffMatrix, RHS, ConcentrationDist, temp_ConcentrationDist, opts,
							   d_x_vec, d_temp_x_vec, d_Coeff, d_RHS, MFL, MFR, D, mesh, &myImg);

		if (opts.verbose == 1)
		{
			printf("Iterations taken = %d\n", iter_taken);
		}
	}

	// non-dimensional and normalized Deff

	myImg.deff = myImg.deff / DCF;

	// Print if applicable

	if (opts.verbose == 1)
	{
		std::cout << "DCF = " << DCF << ", Deff " << myImg.deff << std::endl;
	}

	// create output file

	outputSingle3Phase(opts, mesh, myImg);

	// Create Concentration Map

	if (opts.printCmap == 1)
	{
		createCMAP(ConcentrationDist, &opts, &mesh);
	}

	// Free everything

	unInitializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff);
	free(MFL);
	free(MFR);
	free(CoeffMatrix);
	free(RHS);
	free(ConcentrationDist);
	free(temp_ConcentrationDist);
	free(D);

	return 0;
}

int SingleSim(options opts){
	/*
		Function to read a single image and simulate the effective diffusivity. Results
		 are stored on the output file

		Inputs:
			Datastructure with user-defined simulation options
		Outputs:
			none
	*/

	// Define data structures

	simulationInfo myImg;

	meshInfo mesh;

	// first step is to read the image properly and calculate the porosity

	readImage(opts, &myImg);

	myImg.porosity = calcPorosity(myImg.target_data, myImg.Width, myImg.Height);

	// right now the program only deals with grayscale binary images, so we need to make sure to return that to the user

	if(opts.verbose == 1){
		std::cout << "Width = " << myImg.Width << " Height = " << myImg.Height << " Channel = " << myImg.nChannels << std::endl;
		std::cout << "Porosity = " << myImg.porosity << std::endl;
	}

	if (myImg.nChannels != 1){
		printf("Error: please enter a grascale image with 1 channel.\n Current number of channels = %d\n", myImg.nChannels);
		return 1;
	}

	// Sort out the current mesh

	if(opts.MeshIncreaseX < 1 || opts.MeshIncreaseY < 1){						// Return error if mesh refinement is smaller than 1
		printf("MeshIncrease has to be an integer greater than 1.\n");
		return 1;
	}

	// Define number of cells in each direction

	mesh.numCellsX = myImg.Width*opts.MeshIncreaseX;
	mesh.numCellsY = myImg.Height*opts.MeshIncreaseY;
	mesh.nElements = mesh.numCellsX*mesh.numCellsY;
	mesh.dx = 1.0/mesh.numCellsX;
	mesh.dy = 1.0/mesh.numCellsY;

	// Use pathfinding algorithm

	myImg.PathFlag = false;

	// Declare search boundaries for the domain

	unsigned int *Grid = (unsigned int*)malloc(sizeof(unsigned int)*mesh.numCellsX*mesh.numCellsY);

	for(int i = 0; i<mesh.numCellsY; i++){
		for(int j = 0; j<mesh.numCellsX; j++){
			if(myImg.target_data[i*myImg.Width + j] > 150){
				Grid[i*myImg.Width + j] = 1;
			} else{
				Grid[i*myImg.Width + j] = 0;
			}
		}
	}

	// Search path

	FloodFill(Grid, &mesh, &myImg);

	free(Grid);

	// For this algorithm we continue whether there was a path or not

	// Diffusion coefficients

	double DCF_Max = opts.DCfluid;
	double DCF = 10.0f;
	double DCS = opts.DCsolid;

	// We will use an artificial scaling of the diffusion coefficient to converge to the correct solution

	// Declare useful arrays
	double *D = (double*)malloc(sizeof(double)*mesh.numCellsX*mesh.numCellsY); 			// Grid matrix containing the diffusion coefficient of each cell with appropriate mesh
	double *MFL = (double*)malloc(sizeof(double)*mesh.numCellsY);										// mass flux in the left boundary
	double *MFR = (double*)malloc(sizeof(double)*mesh.numCellsY);										// mass flux in the right boundary

	double *CoeffMatrix = (double *)malloc(sizeof(double)*mesh.nElements*5);					// array will be used to store our coefficient matrix
	double *RHS = (double *)malloc(sizeof(double)*mesh.nElements);										// array used to store RHS of the system of equations
	double *ConcentrationDist = (double *)malloc(sizeof(double)*mesh.nElements);			// array used to store the solution to the system of equations
	double *temp_ConcentrationDist = (double *)malloc(sizeof(double)*mesh.nElements);			// array used to store the solution to the system of equations

	// Initialize the concentration map with a linear gradient between the two boundaries
	for(int i = 0; i<mesh.numCellsY; i++){
		for(int j = 0; j<mesh.numCellsX; j++){
			ConcentrationDist[i*mesh.numCellsX + j] = (double)j/mesh.numCellsX*(opts.CRight - opts.CLeft) + opts.CLeft;
		}
	}

	// Zero the time

	myImg.gpuTime = 0;

	// Declare GPU arrays

	double *d_x_vec = NULL;
	double *d_temp_x_vec = NULL;
	
	double *d_Coeff = NULL;
	double *d_RHS = NULL;

	// Initialize the GPU arrays

	if(!initializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff, mesh))
	{
		printf("\n Error when allocating space in GPU");
		unInitializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff);
		return 0;
	}

	// determine DCF scaling approach

	int count = 1;

	while(DCF <= DCF_Max){
		DCF = std::pow(100,count);
		if(DCF >= DCF_Max){
			DCF = DCF_Max;
		}
		// Populate arrays wiht zeroes
		memset(MFL, 0, sizeof(MFL));
		memset(MFR, 0, sizeof(MFR));
		memset(CoeffMatrix, 0, sizeof(CoeffMatrix));
		memset(RHS, 0, sizeof(RHS));
		// Populate D according to DCF, DCS, and target image. Mesh amplification is employed at this step
		// 			on converting the actual 2D image into a simulation domain.
		for(int i = 0; i<mesh.numCellsY; i++){
			MFL[i] = 0;
			MFR[i] = 0;
			for(int j = 0; j<mesh.numCellsX; j++){
				int targetIndexRow = i/opts.MeshIncreaseY;
				int targetIndexCol = j/opts.MeshIncreaseX;
				if(myImg.target_data[targetIndexRow*myImg.Width + targetIndexCol] < 150){
					D[i*mesh.numCellsX + j] = DCF;
				} else{
					D[i*mesh.numCellsX + j] = DCS;
				}
			}
		}

		// Now that we have all pieces, generate the coefficient matrix

		DiscretizeMatrix2D(D, CoeffMatrix, RHS, mesh, opts);

		// Solve with GPU
		int iter_taken = 0;
		iter_taken = JacobiGPU(CoeffMatrix, RHS, ConcentrationDist, temp_ConcentrationDist, opts, 
			d_x_vec, d_temp_x_vec, d_Coeff, d_RHS, MFL, MFR, D, mesh, &myImg);

		if(opts.verbose == 1){
			printf("Iterations taken = %d\n", iter_taken);
		}

		// non-dimensional and normalized Deff

		myImg.deff = myImg.deff/DCF;

		// Print if applicable

		if(opts.verbose == 1){
			std::cout << "DCF = " << DCF << ", Deff " << myImg.deff << std::endl;
		}

		// update DCF

		if(DCF == DCF_Max){
			break;
		}

		count++;
	}

	// create output file 

	outputSingle(opts, mesh, myImg);

	// Create Concentration Map

	if(opts.printCmap == 1){
		createCMAP(ConcentrationDist, &opts, &mesh);
	}

	// Free everything

	unInitializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff);
	free(MFL);
	free(MFR);
	free(CoeffMatrix);
	free(RHS);
	free(ConcentrationDist);
	free(temp_ConcentrationDist);
	free(D);

	return 0;
}

int BatchSim(options opts){
	/*
		Function to read a batch of images and simulate the effective diffusivity. Results
		 are stored on the output file.

		Inputs:
			Datastructure with user-defined simulation options
		Outputs:
			none
	*/

	// Start from image 0

	int imageNum = 0;

	// array to store image name

	char imageName[100];

	// Create array to store all outputs
	// In order: imgNum, porosity,PathFlag,Deff,Time,nElements,converge,ds,df

	double *output = (double *)malloc(sizeof(double)*opts.NumImg*9);

	while(imageNum < opts.NumImg){
		// Define data structures

		simulationInfo myImg;

		meshInfo mesh;

		// Image name

		sprintf(imageName,"%05d.jpg",imageNum);

		// Read image

		readImageBatch(opts, &myImg, imageName);

		// Calculate porosity

		myImg.porosity = calcPorosity(myImg.target_data, myImg.Width, myImg.Height);

		// right now the program only deals with grayscale binary images, so we need to make sure to return that to the user

		if(opts.verbose == 1){
			std::cout << "Width = " << myImg.Width << " Height = " << myImg.Height << " Channel = " << myImg.nChannels << std::endl;
			std::cout << "Porosity = " << myImg.porosity << std::endl;
		}

		if (myImg.nChannels != 1){
			printf("Error: please enter a grascale image with 1 channel.\n Current number of channels = %d\n", myImg.nChannels);
			return 1;
		}

		// Sort out the current mesh

		if(opts.MeshIncreaseX < 1 || opts.MeshIncreaseY < 1){						// Return error if mesh refinement is smaller than 1
			printf("MeshIncrease has to be an integer greater than 1.\n");
			return 1;
		}

		// Define number of cells in each direction

		mesh.numCellsX = myImg.Width*opts.MeshIncreaseX;
		mesh.numCellsY = myImg.Height*opts.MeshIncreaseY;
		mesh.nElements = mesh.numCellsX*mesh.numCellsY;
		mesh.dx = 1.0/mesh.numCellsX;
		mesh.dy = 1.0/mesh.numCellsY;

		// Use pathfinding algorithm

		myImg.PathFlag = false;

		// Declare search boundaries for the domain

		unsigned int *Grid = (unsigned int*)malloc(sizeof(unsigned int)*mesh.numCellsX*mesh.numCellsY);

		for(int i = 0; i<mesh.numCellsY; i++){
			for(int j = 0; j<mesh.numCellsX; j++){
				if(myImg.target_data[i*myImg.Width + j] > 150){
					Grid[i*myImg.Width + j] = 1;
				} else{
					Grid[i*myImg.Width + j] = 0;
				}
			}
		}

		// Search path

		FloodFill(Grid, &mesh, &myImg);

		// For this algorithm we continue whether there was a path or not

		// Diffusion coefficients

		double DCF = opts.DCfluid;
		double DCS = opts.DCsolid;

		// We will use an artificial scaling of the diffusion coefficient to converge to the correct solution

		// Declare useful arrays
		double *D = (double *)malloc(sizeof(double)*mesh.numCellsX*mesh.numCellsY); 			// Grid matrix containing the diffusion coefficient of each cell with appropriate mesh
		double *MFL = (double *)malloc(sizeof(double)*mesh.numCellsY);										// mass flux in the left boundary
		double *MFR = (double *)malloc(sizeof(double)*mesh.numCellsY);										// mass flux in the right boundary

		double *CoeffMatrix = (double *)malloc(sizeof(double)*mesh.nElements*5);					// array will be used to store our coefficient matrix
		double *RHS = (double *)malloc(sizeof(double)*mesh.nElements);										// array used to store RHS of the system of equations
		double *ConcentrationDist = (double *)malloc(sizeof(double)*mesh.nElements);			// array used to store the solution to the system of equations
		double *temp_ConcentrationDist = (double *)malloc(sizeof(double)*mesh.nElements);			// array used to store the solution to the system of equations

		// Initialize the concentration map with a linear gradient between the two boundaries
		for(int i = 0; i<mesh.numCellsY; i++){
			for(int j = 0; j<mesh.numCellsX; j++){
				ConcentrationDist[i*mesh.numCellsX + j] = (double)j/mesh.numCellsX*(opts.CRight - opts.CLeft) + opts.CLeft;
			}
		}

		// Zero the time

		myImg.gpuTime = 0;

		// Declare GPU arrays

		double *d_x_vec = NULL;
		double *d_temp_x_vec = NULL;
		
		double *d_Coeff = NULL;
		double *d_RHS = NULL;

		// Initialize the GPU arrays

		if(!initializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff, mesh))
		{
			printf("\n Error when allocating space in GPU");
			unInitializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff);
			return 0;
		}

		// Populate arrays wiht zeroes
		memset(MFL, 0, sizeof(MFL));
		memset(MFR, 0, sizeof(MFR));
		memset(CoeffMatrix, 0, sizeof(CoeffMatrix));
		memset(RHS, 0, sizeof(RHS));
		// Populate D according to DCF, DCS, and target image. Mesh amplification is employed at this step
		for(int i = 0; i<mesh.numCellsY; i++){
			MFL[i] = 0;
			MFR[i] = 0;
			for(int j = 0; j<mesh.numCellsX; j++){
				int targetIndexRow = i/opts.MeshIncreaseY;
				int targetIndexCol = j/opts.MeshIncreaseX;
				if(myImg.target_data[targetIndexRow*myImg.Width + targetIndexCol] < 150){
					D[i*mesh.numCellsX + j] = DCF;
				} else{
					D[i*mesh.numCellsX + j] = DCS;
				}
			}
		}

		// Discretize

		DiscretizeMatrix2D(D, CoeffMatrix, RHS, mesh, opts);

		// Solve with GPU
		int iter_taken = 0;
		iter_taken = JacobiGPU(CoeffMatrix, RHS, ConcentrationDist, temp_ConcentrationDist, opts, 
			d_x_vec, d_temp_x_vec, d_Coeff, d_RHS, MFL, MFR, D, mesh, &myImg);

		if(opts.verbose == 1){
			printf("Iterations taken = %d\n", iter_taken);
		}

		// Normalize

		myImg.deff = myImg.deff/DCF;

		if(opts.verbose == 1){
			std::cout << "Number" << imageNum << "DCF = " << DCF << ", Deff " << myImg.deff << std::endl;
		}

		// save output
		// imgNum, porosity,PathFlag,Deff,Time,nElements,converge,ds,df

		output[imageNum*9 + 0] = (double) imageNum;
		output[imageNum*9 + 1] = myImg.porosity;
		output[imageNum*9 + 2] = (double) myImg.PathFlag;
		output[imageNum*9 + 3] = myImg.deff;
		output[imageNum*9 + 4] = myImg.gpuTime/1000;
		output[imageNum*9 + 5] = mesh.nElements;
		output[imageNum*9 + 6] = myImg.conv;
		output[imageNum*9 + 7] = opts.DCsolid;
		output[imageNum*9 + 8] = DCF;

		// Free everything

		unInitializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff);
		free(MFL);
		free(MFR);
		free(CoeffMatrix);
		free(RHS);
		free(ConcentrationDist);
		free(temp_ConcentrationDist);
		free(D);

		imageNum++;

	}

	outputBatch(opts, output);

	return 0;
}

int BatchSim3Phase(options opts){
	/*
		Function to read a batch of images and simulate the effective diffusivity. Results
		 are stored on the output file.

		Inputs:
			Datastructure with user-defined simulation options
		Outputs:
			none
	*/

	// Start from image 0

	int imageNum = 0;

	// array to store image name

	char imageName[100];
	char CMapName[100];

	// Create array to store all outputs
	// In order: imgNum,SVF,LVF,PathFlag,Deff,Time,nElements,converge,ds,df

	double *output = (double *)malloc(sizeof(double)*opts.NumImg*10);

	while(imageNum < opts.NumImg){
		// Define data structures

		simulationInfo myImg;

		meshInfo mesh;

		// Image name

		sprintf(imageName,"%05d.jpg",imageNum);

		// Read image

		readImageBatch(opts, &myImg, imageName);

		// right now the program only deals with grayscale binary images, so we need to make sure to return that to the user

		if(opts.verbose == 1){
			std::cout << imageName <<std::endl;
			std::cout << "Width = " << myImg.Width << " Height = " << myImg.Height << " Channel = " << myImg.nChannels << std::endl;
		}

		if (myImg.nChannels != 1){
			printf("Error: please enter a grascale image with 1 channel.\n Current number of channels = %d\n", myImg.nChannels);
			return 1;
		}

		// Sort out the current mesh

		if(opts.MeshIncreaseX < 1 || opts.MeshIncreaseY < 1){						// Return error if mesh refinement is smaller than 1
			printf("MeshIncrease has to be an integer greater than 1.\n");
			return 1;
		}

		// Define number of cells in each direction

		mesh.numCellsX = myImg.Width*opts.MeshIncreaseX;
		mesh.numCellsY = myImg.Height*opts.MeshIncreaseY;
		mesh.nElements = mesh.numCellsX*mesh.numCellsY;
		mesh.dx = 1.0/mesh.numCellsX;
		mesh.dy = 1.0/mesh.numCellsY;

		// Use pathfinding algorithm

		myImg.PathFlag = false;

		// Declare search boundaries for the domain

		unsigned int *Grid = (unsigned int*)malloc(sizeof(unsigned int)*mesh.numCellsX*mesh.numCellsY);

		for(int i = 0; i<mesh.numCellsY; i++){
			for(int j = 0; j<mesh.numCellsX; j++){
				if(myImg.target_data[i*myImg.Width + j] > 150){
					Grid[i*myImg.Width + j] = 1;
				} else{
					Grid[i*myImg.Width + j] = 0;
				}
			}
		}

		// Search path

		FloodFill(Grid, &mesh, &myImg);

		// For this algorithm we continue whether there was a path or not

		// Diffusion coefficients

		double DCF = opts.DCfluid;
		double DCG = opts.DCgas;
		double DCS = opts.DCsolid;

		// Declare useful arrays
		double *D = (double *)malloc(sizeof(double) * mesh.numCellsX * mesh.numCellsY); // Grid matrix containing the diffusion coefficient of each cell with appropriate mesh
		double *MFL = (double *)malloc(sizeof(double) * mesh.numCellsY);				// mass flux in the left boundary
		double *MFR = (double *)malloc(sizeof(double) * mesh.numCellsY);				// mass flux in the right boundary

		double *CoeffMatrix = (double *)malloc(sizeof(double) * mesh.nElements * 5);		// array will be used to store our coefficient matrix
		double *RHS = (double *)malloc(sizeof(double) * mesh.nElements);					// array used to store RHS of the system of equations
		double *ConcentrationDist = (double *)malloc(sizeof(double) * mesh.nElements);		// array used to store the solution to the system of equations
		double *temp_ConcentrationDist = (double *)malloc(sizeof(double) * mesh.nElements); // array used to store the solution to the system of equations

		// Initialize the concentration map with a linear gradient between the two boundaries
		for (int i = 0; i < mesh.numCellsY; i++)
		{
			for (int j = 0; j < mesh.numCellsX; j++)
			{
				ConcentrationDist[i * mesh.numCellsX + j] = (double)j / mesh.numCellsX * (opts.CRight - opts.CLeft) + opts.CLeft;
			}
		}

		// Zero the time

		myImg.gpuTime = 0;

		// Declare GPU arrays

		double *d_x_vec = NULL;
		double *d_temp_x_vec = NULL;

		double *d_Coeff = NULL;
		double *d_RHS = NULL;

		// Initialize the GPU arrays

		if (!initializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff, mesh))
		{
			printf("\n Error when allocating space in GPU");
			unInitializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff);
			return 0;
		}

		// Populate D according to DCF, DCS, DCG, and target image. Mesh amplification is employed at this step
		// 	on converting the actual 2D image into a simulation domain.

		/*

		Target Grayscale Image Requirements:
		- Solid = 255
		- Fluid = 150
		- Gas = 0

		*/

		bool preCond = true;

		if (preCond == false)
		{
			// No preconditioning necessary, proceed normally
			for (int i = 0; i < mesh.numCellsY; i++)
			{
				MFL[i] = 0;
				MFR[i] = 0;
				for (int j = 0; j < mesh.numCellsX; j++)
				{
					int targetIndexRow = i / opts.MeshIncreaseY;
					int targetIndexCol = j / opts.MeshIncreaseX;
					if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] > 200)
					{
						D[i * mesh.numCellsX + j] = DCS;
					}
					else if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] < 50)
					{
						D[i * mesh.numCellsX + j] = DCG;
					}
					else
					{
						D[i * mesh.numCellsX + j] = DCF;
					}
				}
			}

			// Calculate phase fractions

			calcFracts3D(&myImg, D, &mesh, &opts);

			// Now that we have all pieces, generate the coefficient matrix

			DiscretizeMatrix2D_ImpSolid(D, CoeffMatrix, RHS, mesh, opts, Grid);

			// Solve with GPU
			int iter_taken = 0;
			iter_taken = JacobiGPU(CoeffMatrix, RHS, ConcentrationDist, temp_ConcentrationDist, opts,
								   d_x_vec, d_temp_x_vec, d_Coeff, d_RHS, MFL, MFR, D, mesh, &myImg);

			if (opts.verbose == 1)
			{
				printf("Iterations taken = %d\n", iter_taken);
			}
		}
		else
		{
			// Pre-condition, then solve
			double DCG_Temp = 10;
			int preCondStage = 1;

			// save original settings
			double originalTol = opts.ConvergeCriteria;
			double originalMaxIter = opts.MAX_ITER;

			// Decrease the strictness of convergence for the pre-conditioner

			opts.ConvergeCriteria = originalTol * 10;
			opts.MAX_ITER = 1e6;

			while (DCG_Temp < DCG)
			{
				if (opts.verbose == 1)
				{
					printf("Pre-Cond Stage %d: DCG = %1.3e\n", preCondStage, DCG_Temp);
				}
				for (int i = 0; i < mesh.numCellsY; i++)
				{
					MFL[i] = 0;
					MFR[i] = 0;
					for (int j = 0; j < mesh.numCellsX; j++)
					{
						int targetIndexRow = i / opts.MeshIncreaseY;
						int targetIndexCol = j / opts.MeshIncreaseX;
						if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] > 200)
						{
							D[i * mesh.numCellsX + j] = DCS;
						}
						else if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] < 50)
						{
							D[i * mesh.numCellsX + j] = DCG_Temp;
						}
						else
						{
							D[i * mesh.numCellsX + j] = DCF;
						}
					}
				}

				// Now that we have all pieces, generate the coefficient matrix

				DiscretizeMatrix2D_ImpSolid(D, CoeffMatrix, RHS, mesh, opts, Grid);

				// Solve with GPU
				int iter_taken = 0;
				iter_taken = JacobiGPUPreCond(CoeffMatrix, RHS, ConcentrationDist, temp_ConcentrationDist, opts,
											  d_x_vec, d_temp_x_vec, d_Coeff, d_RHS, MFL, MFR, D, mesh, &myImg);

				if (opts.verbose == 1)
				{
					printf("Iterations taken = %d\n", iter_taken);
				}

				DCG_Temp = DCG_Temp * 10;
				preCondStage++;
			}

			// Pre-Conditioning done, solve actual system

			opts.ConvergeCriteria = originalTol;
			opts.MAX_ITER = originalMaxIter;

			// No preconditioning necessary, proceed normally
			for (int i = 0; i < mesh.numCellsY; i++)
			{
				MFL[i] = 0;
				MFR[i] = 0;
				for (int j = 0; j < mesh.numCellsX; j++)
				{
					int targetIndexRow = i / opts.MeshIncreaseY;
					int targetIndexCol = j / opts.MeshIncreaseX;
					if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] > 200)
					{
						D[i * mesh.numCellsX + j] = DCS;
					}
					else if (myImg.target_data[targetIndexRow * myImg.Width + targetIndexCol] < 50)
					{
						D[i * mesh.numCellsX + j] = DCG;
					}
					else
					{
						D[i * mesh.numCellsX + j] = DCF;
					}
				}
			}

			// Calculate phase fractions

			calcFracts3D(&myImg, D, &mesh, &opts);

			// Now that we have all pieces, generate the coefficient matrix

			DiscretizeMatrix2D_ImpSolid(D, CoeffMatrix, RHS, mesh, opts, Grid);

			// Solve with GPU
			int iter_taken = 0;
			iter_taken = JacobiGPU(CoeffMatrix, RHS, ConcentrationDist, temp_ConcentrationDist, opts,
								   d_x_vec, d_temp_x_vec, d_Coeff, d_RHS, MFL, MFR, D, mesh, &myImg);

			if (opts.verbose == 1)
			{
				printf("Iterations taken = %d\n", iter_taken);
			}
		}

		// non-dimensional and normalized Deff

		myImg.deff = myImg.deff / DCF;

		// Print if applicable

		if (opts.verbose == 1)
		{
			std::cout << "DCF = " << DCF << ", Deff " << myImg.deff << std::endl;
		}

		// save output
		// imgNum, porosity,PathFlag,Deff,Time,nElements,converge,ds,df

		output[imageNum * 10 + 0] = (double)imageNum;
		output[imageNum * 10 + 1] = myImg.SVF;
		output[imageNum * 10 + 2] = myImg.LVF;
		output[imageNum * 10 + 3] = (double)myImg.PathFlag;
		output[imageNum * 10 + 4] = myImg.deff;
		output[imageNum * 10 + 5] = myImg.gpuTime / 1000;
		output[imageNum * 10 + 6] = mesh.nElements;
		output[imageNum * 10 + 7] = myImg.conv;
		output[imageNum * 10 + 8] = opts.DCsolid;
		output[imageNum * 10 + 9] = DCF;

		// Save concentration maps if applicable

		if (opts.printCmap == 1){
			sprintf(CMapName,"CMAP_%05d.csv",imageNum);
			createCMAPBatch(ConcentrationDist, CMapName, &mesh);
		}

		// Free everything

		unInitializeGPU(&d_x_vec, &d_temp_x_vec, &d_RHS, &d_Coeff);
		free(MFL);
		free(MFR);
		free(CoeffMatrix);
		free(Grid);
		free(RHS);
		free(ConcentrationDist);
		free(temp_ConcentrationDist);
		free(D);

		imageNum++;

	}

	outputBatch3Phase(opts, output);

	return 0;
}