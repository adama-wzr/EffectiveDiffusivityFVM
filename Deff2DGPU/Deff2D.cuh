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


typedef struct{
	int Width;
	int Height;
	int nChannels;
	float porosity;
	float gpuTime;
	unsigned char *target_data;
	float deff;
	bool PathFlag;
	float conv;
}simulationInfo;


typedef struct{
	int numCellsX;
	int numCellsY;
	int nElements;
	float dx;
	float dy;
}meshInfo;

//Data structures for a* algorithm:

// generate structure to store global information about the domain

typedef struct{
	unsigned int xSize;
	unsigned int ySize;
	bool verbose;
}domainInfo;

// node struct will hold information on cell parent cells, f, g, and h.

typedef struct{
	int parentRow, parentCol;
	float f,g,h;
}node;

// define pair for coords

typedef std::pair<int, int> coordPair;

// define pair <float, pair<i,j>> for open list

typedef std::pair<float, std::pair<int,int> > OpenListInfo;

// GPU Jacobi-Iteration Kernel

__global__ void updateX_V1(float* A, float* x, float* b, float* xNew, meshInfo mesh)
{
	unsigned int myRow = blockIdx.x * blockDim.x + threadIdx.x;

	if (myRow < mesh.nElements){
		float sigma = 0;
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


float calcPorosity(unsigned char* imageAddress, int Width, int Height){
	/*
		calcPorosity
		Inputs:
			- imageAddress: pointer to the read image.
			- Width: original width from std_image
			- Height: original height from std_image

		Output:
			- porosity: float containing porosity.

		Function calculates porosity by counting pixels.
	*/

	float totalCells = (float)Height*Width;
	float porosity = 0;
	for(int i = 0; i<Height; i++){
		for(int j = 0; j<Width; j++){
			if(imageAddress[i*Width + j] < 150){
				porosity += 1.0/totalCells;
			}
		}
	}

	return porosity;
}

int aStarMain(unsigned int* GRID, domainInfo info){
	/*
		aStarMain Function
		Inputs:
			- unsigned int GRID: grid, at each location either a 1 or a 0.
				1 means solid, 0 void. Those are the boundary conditions.
			- domainInfo info: data structure with info regarding height and width of the domain
		Output:
			- either a one or a zero. One means there is a path, zero means there isn't.
	*/

	// Initialize both lists, open and closed as arrays

	bool* closedList = (bool *)malloc(sizeof(bool)*info.xSize*info.ySize);

	memset(closedList, false, sizeof(closedList));

	// Declare 2D array of structure type "node"
	// Node contains information such as parent coordinates, g, h, and f

	node nodeInfo[info.ySize][info.ySize];

	// Initialize all paremeters

	for(int i = 0; i<info.ySize; i++){
		for(int j = 0; j<info.xSize; j++){
			nodeInfo[i][j].f = FLT_MAX;
			nodeInfo[i][j].g = FLT_MAX;
			nodeInfo[i][j].h = FLT_MAX;
			nodeInfo[i][j].parentCol = -1;
			nodeInfo[i][j].parentRow = -1;
		}
	}

	// Initialize parameters for all starting nodes

	for(int i = 0; i<info.ySize; i++){
		if(GRID[i*info.xSize + 0] == 0){
			nodeInfo[i][0].f = 0.0;
			nodeInfo[i][0].g = 0.0;
			nodeInfo[i][0].h = 0.0;
			nodeInfo[i][0].parentCol = 0;
			nodeInfo[i][0].parentRow = i;
		}
	}

	// Create open list

	std::set<OpenListInfo> openList;

	// Insert all starting nodes into the open list

	for(int i = 0; i<info.ySize; i++){
		openList.insert(std::make_pair(0.0, std::make_pair(i,0)));
	}

	// set destination flag to false

	bool foundDest = false;

	// begin loop to find path. If openList is empty, terminate the loop

	while(!openList.empty()){
		// First step is to pop the fist entry on the list
		OpenListInfo pop = *openList.begin();

		// remove from open list
		openList.erase(openList.begin());

		// Add to the closed list
		int row = pop.second.first; // first argument of the second pair
		int col = pop.second.second; // second argument of second pair
		closedList[row*info.xSize + col] = true;

		/*
			Now we need to generate all 4 successors from the popped cell.
			The successors are north, south, east, and west.
			
			North index = i - 1, j
			South index = i + 1, j
			East index =  i    , j + 1
			West index =  i    , j - 1
		*/
		float gNew, hNew, fNew;

		// Evaluate North
		
		int tempCol = col;
		int tempRow = row;

		// adjust North for periodic boundary condition

		if(row == 0){
			tempRow = info.ySize - 1;
		} else{
			tempRow = row - 1;
		}

		// check if we reached destination, which is the entire right boundary
		if(tempCol == info.ySize - 1 && GRID[tempRow*info.xSize + tempCol] != 1){
			nodeInfo[tempRow][tempCol].parentRow = row;
			nodeInfo[tempRow][tempCol].parentCol = col;
			if(info.verbose == true){
				printf("Path found.\n");
			}
			// Found dest, update flag and terminate
			foundDest = true;
			return foundDest;
		} else if(closedList[tempRow*info.xSize + tempCol] == false && GRID[tempRow*info.xSize + tempCol] == 0) // check if successor is not on closed list and not a solid wall
		{
			gNew = nodeInfo[row][col].g + 1.0;	// cost from moving from last cell to this cell
			hNew = (info.xSize - 1) - tempCol; // Since entire right boundary is the distance, h is just a count of the number of columns from the right.	
			fNew = gNew + hNew;					// total cost is just h+g
			// Check if on open list. If yes, update f,g, and h accordingly.
			// If not, add it to open list.
			if(nodeInfo[tempRow][tempCol].f == FLT_MAX || nodeInfo[tempRow][tempCol].f > fNew){
				openList.insert(std::make_pair(fNew, std::make_pair(tempRow, tempCol)));
				nodeInfo[tempRow][tempCol].f = fNew;
				nodeInfo[tempRow][tempCol].g = gNew;
				nodeInfo[tempRow][tempCol].h = hNew;
				nodeInfo[tempRow][tempCol].parentRow = row;
				nodeInfo[tempRow][tempCol].parentCol = col;
			}
		}
			

		// Evaluate South

		tempCol = col;
		tempRow = row;

		// Adjust for periodic BC

		if(row == info.ySize - 1){
			tempRow = 0;
		} else{
			tempRow = row + 1;
		}

		// check if we reached destination, which is the entire right boundary
		if(tempCol == info.ySize - 1 && GRID[tempRow*info.xSize + tempCol] != 1){
			nodeInfo[tempRow][tempCol].parentRow = row;
			nodeInfo[tempRow][tempCol].parentCol = col;
			if(info.verbose == true){
				printf("Path found.\n");
			}
			// Found dest, update flag and terminate
			foundDest = true;
			return foundDest;
		} else if(closedList[tempRow*info.xSize + tempCol] == false && GRID[tempRow*info.xSize + tempCol] == 0) // check if successor is not on closed list and not a solid wall
		{
			gNew = nodeInfo[row][col].g + 1.0;	// cost from moving from last cell to this cell
			hNew = (info.xSize - 1) - tempCol; // Since entire right boundary is the distance, h is just a count of the number of columns from the right.	
			fNew = gNew + hNew;					// total cost is just h+g
			// Check if on open list. If yes, update f,g, and h accordingly.
			// If not, add it to open list.
			if(nodeInfo[tempRow][tempCol].f == FLT_MAX || nodeInfo[tempRow][tempCol].f > fNew){
				openList.insert(std::make_pair(fNew, std::make_pair(tempRow, tempCol)));
				nodeInfo[tempRow][tempCol].f = fNew;
				nodeInfo[tempRow][tempCol].g = gNew;
				nodeInfo[tempRow][tempCol].h = hNew;
				nodeInfo[tempRow][tempCol].parentRow = row;
				nodeInfo[tempRow][tempCol].parentCol = col;
			}
		}

		// Evaluate East (if it exists)

		if(col != info.xSize - 1){
			tempRow = row;
			tempCol = col + 1;

			// check if we reached destination, which is the entire right boundary
			if(tempCol == info.ySize - 1 && GRID[tempRow*info.xSize + tempCol] != 1){
				nodeInfo[tempRow][tempCol].parentRow = row;
				nodeInfo[tempRow][tempCol].parentCol = col;
				if(info.verbose == true){
					printf("Path found.\n");
				}
				// Found dest, update flag and terminate
				foundDest = true;
				return foundDest;
			} else if(closedList[tempRow*info.xSize + tempCol] == false && GRID[tempRow*info.xSize + tempCol] == 0) // check if successor is not on closed list and not a solid wall
			{
				gNew = nodeInfo[row][col].g + 1.0;	// cost from moving from last cell to this cell
				hNew = (info.xSize - 1) - tempCol; // Since entire right boundary is the distance, h is just a count of the number of columns from the right.	
				fNew = gNew + hNew;					// total cost is just h+g
				// Check if on open list. If yes, update f,g, and h accordingly.
				// If not, add it to open list.
				if(nodeInfo[tempRow][tempCol].f == FLT_MAX || nodeInfo[tempRow][tempCol].f > fNew){
					openList.insert(std::make_pair(fNew, std::make_pair(tempRow, tempCol)));
					nodeInfo[tempRow][tempCol].f = fNew;
					nodeInfo[tempRow][tempCol].g = gNew;
					nodeInfo[tempRow][tempCol].h = hNew;
					nodeInfo[tempRow][tempCol].parentRow = row;
					nodeInfo[tempRow][tempCol].parentCol = col;
				}
			}
		}

		// Evaluate West

		if(col != 0){
			tempRow = row;
			tempCol = col;

			// check if we reached destination, which is the entire right boundary
			if(tempCol == info.ySize - 1 && GRID[tempRow*info.xSize + tempCol] != 1){
				nodeInfo[tempRow][tempCol].parentRow = row;
				nodeInfo[tempRow][tempCol].parentCol = col;
				if(info.verbose == true){
					printf("Path found.\n");
				}
				// Found dest, update flag and terminate
				foundDest = true;
				return foundDest;
			} else if(closedList[tempRow*info.xSize + tempCol] == false && GRID[tempRow*info.xSize + tempCol] == 0) // check if successor is not on closed list and not a solid wall
			{
				gNew = nodeInfo[row][col].g + 1.0;	// cost from moving from last cell to this cell
				hNew = (info.xSize - 1) - tempCol; // Since entire right boundary is the distance, h is just a count of the number of columns from the right.	
				fNew = gNew + hNew;					// total cost is just h+g
				// Check if on open list. If yes, update f,g, and h accordingly.
				// If not, add it to open list.
				if(nodeInfo[tempRow][tempCol].f == FLT_MAX || nodeInfo[tempRow][tempCol].f > fNew){
					openList.insert(std::make_pair(fNew, std::make_pair(tempRow, tempCol)));
					nodeInfo[tempRow][tempCol].f = fNew;
					nodeInfo[tempRow][tempCol].g = gNew;
					nodeInfo[tempRow][tempCol].h = hNew;
					nodeInfo[tempRow][tempCol].parentRow = row;
					nodeInfo[tempRow][tempCol].parentCol = col;
				}
			}
		}
	}
	if(info.verbose == true){
		printf("Failed to find a path.\n");
	}
	return foundDest;

}

float WeightedHarmonicMean(float w1, float w2, float x1, float x2){
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
	float H = (w1 + w2)/(w1/x1 + w2/x2);
	return H;
}

int DiscretizeMatrix2D(float* D, float* A, float* b, meshInfo mesh, options opts){
	/*
		DiscretizeMatrix2D

		Inputs:
			- pointer to float array D, where local diffusion coefficients are stored
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
	float dxw, dxe, dys, dyn;
	float kw, ke, ks, kn;

	float dx, dy;
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

int initializeGPU(float **d_x_vec, float **d_temp_x_vec, float **d_RHS, float **d_Coeff, meshInfo mesh){

	// Set device, when cudaStatus is called give status of assigned device.
	// This is important to know if we are running out of GPU space
	cudaError_t cudaStatus = cudaSetDevice(0);

	// Start by allocating space in GPU memory

	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		getchar();
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&(*d_x_vec), mesh.nElements*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		getchar();
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&(*d_temp_x_vec), mesh.nElements*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		getchar();
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&(*d_RHS), mesh.nElements*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		getchar();
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&(*d_Coeff), mesh.nElements*sizeof(float)*5);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		getchar();
        return 0;
    }

    // Set GPU buffers (initializing matrices to 0)

     // Memset GPU buffers
    cudaStatus = cudaMemset((*d_x_vec),0, mesh.nElements*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
		getchar();
        return 0;
    }

	// Memset GPU buffers
    cudaStatus = cudaMemset((*d_temp_x_vec),0, mesh.nElements*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
		getchar();
        return 0;
    }

     // Memset GPU buffers
    cudaStatus = cudaMemset((*d_RHS),0, mesh.nElements*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
		getchar();
        return 0;
    }

	// Memset GPU buffers
    cudaStatus = cudaMemset((*d_Coeff),0, 5*mesh.nElements*sizeof(float));		// coefficient matrix has the 5 main diagonals for all elements
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
		getchar();
        return 0;
    }

    return 1;
}

void unInitializeGPU(float **d_x_vec, float **d_temp_x_vec, float **d_RHS, float **d_Coeff)
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

int JacobiGPU(float *arr, float *sol, float *x_vec, float *temp_x_vec, options opts,
	float *d_x_vec, float *d_temp_x_vec, float *d_Coeff, float *d_RHS, float *MFL, float *MFR, float *D, meshInfo mesh, simulationInfo* myImg)
{

	int iterCount = 0;
	float percentChange = 1;
	int threads_per_block = 160;
	int numBlocks = mesh.nElements/threads_per_block + 1;
	float deffOld = 1;
	float deffNew = 1;
	int iterToCheck = 1000;
	float Q1,Q2;
	float qAvg = 0;
	float dx,dy;
	int numRows = mesh.numCellsY;
	int numCols = mesh.numCellsX;
	const char *str = (char*) malloc(1024); // To store error string

	dx = mesh.dx;
	dy = mesh.dy;

	int nRows = mesh.nElements;	// number of rows in the coefficient matrix
	int nCols = 5;							// number of cols in the coefficient matrix

	// Initialize temp_x_vec

	for(int i = 0; i<nRows; i++){
		temp_x_vec[i] = x_vec[i];
	}

	//Copy arrays into GPU memory

	cudaError_t cudaStatus = cudaMemcpy(d_temp_x_vec, temp_x_vec, sizeof(float) * nRows, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "temp_x_vec cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}
	cudaStatus = cudaMemcpy(d_RHS, sol, sizeof(float)*nRows, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "d_RHS cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}
	cudaStatus = cudaMemcpy(d_Coeff, arr, sizeof(float)*nRows*nCols, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "d_Coeff cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}

	// Declare event to get time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	while(iterCount < opts.MAX_ITER && opts.ConvergeCriteria < percentChange)
	{
		// Call Kernel to Calculate new x-vector
		
		updateX_V1<<<numBlocks, threads_per_block>>>(d_Coeff, d_temp_x_vec, d_RHS, d_x_vec, mesh);

		// update x vector

		d_temp_x_vec = d_x_vec;

		// Convergence related material

		if (iterCount % iterToCheck == 0){
			cudaStatus = cudaMemcpy(x_vec, d_x_vec, sizeof(float) * nRows, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "x_vec cudaMemcpy failed!");
				str = cudaGetErrorString(cudaStatus);
				fprintf(stderr, "CUDA Error!:: %s\n", str);
			}
			Q1 = 0;
			Q2 = 0;
			for (int j = 0; j<numRows; j++){
				MFL[j] = D[j*numRows]*dy*(x_vec[j*numCols] - opts.CLeft)/(dx/2);
				MFR[j] = D[(j + 1)*numRows - 1]*dy*(opts.CRight - x_vec[(j+1)*numCols -1])/(dx/2);
				// printf("T(0,%d) = %2.3f\n", j, x_vec[j*numCols]);
				Q1 += MFL[j];
				Q2 += MFR[j];
			}
			Q1 = Q1;
			Q2 = Q2;
			qAvg = (Q1 + Q2)/2;
			deffNew = qAvg/((opts.CRight - opts.CLeft));
			percentChange = fabs((deffNew - deffOld)/deffOld);
			deffOld = deffNew;

			// printf("Iteration = %d, Keff = %2.3f\n", iterCount, deffNew);

			if (percentChange < 0.001){
				iterToCheck = 100;
			} else if(percentChange < 0.0001){
				iterToCheck = 10;
			}
			myImg->conv = percentChange;
		}

		// Update iteration count
		iterCount++;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaStatus = cudaMemcpy(x_vec, d_x_vec, sizeof(float)*nRows, cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "x_vec cudaMemcpy failed!");
		str = cudaGetErrorString(cudaStatus);
		fprintf(stderr, "CUDA Error!:: %s\n", str);
	}

	myImg->deff = deffNew;

	myImg->gpuTime += elapsedTime;


	return iterCount;
}

int outputSingle(options opts, meshInfo mesh, simulationInfo myImg){
	FILE *OUTPUT;

  OUTPUT = fopen(opts.outputFilename, "a+");
  fprintf(OUTPUT,"imgNum,porosity,PathFlag,Deff,Time,nElements,converge\n");
  fprintf(OUTPUT, "%s,%f,%d,%f,%f,%d,%f\n", opts.inputFilename, myImg.porosity, myImg.PathFlag, myImg.deff, myImg.gpuTime/1000, mesh.nElements, myImg.conv);
  fclose(OUTPUT);
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

	domainInfo info;

	info.xSize = myImg.Width;
	info.ySize = myImg.Height;
	info.verbose = opts.verbose;

	// Declare search boundaries for the domain

	unsigned int *Grid = (unsigned int*)malloc(sizeof(unsigned int)*info.xSize*info.ySize);

	for(int i = 0; i<info.ySize; i++){
		for(int j = 0; j<info.xSize; j++){
			if(myImg.target_data[i*myImg.Width + j] > 150){
				Grid[i*myImg.Width + j] = 1;
			} else{
				Grid[i*myImg.Width + j] = 0;
			}
		}
	}

	// Search path

	myImg.PathFlag = aStarMain(Grid, info);

	free(Grid);

	// For this algorithm we continue whether there was a path or not

	// Diffusion coefficients

	float DCF_Max = opts.DCfluid;
	float DCF = 10.0f;
	float DCS = opts.DCsolid;

	// We will use an artificial scaling of the diffusion coefficient to converge to the correct solution

	// Declare useful arrays
	float *D = (float*)malloc(sizeof(float)*mesh.numCellsX*mesh.numCellsY); 			// Grid matrix containing the diffusion coefficient of each cell with appropriate mesh
	float *MFL = (float*)malloc(sizeof(float)*mesh.numCellsY);										// mass flux in the left boundary
	float *MFR = (float*)malloc(sizeof(float)*mesh.numCellsY);										// mass flux in the right boundary

	float *CoeffMatrix = (float *)malloc(sizeof(float)*mesh.nElements*5);					// array will be used to store our coefficient matrix
	float *RHS = (float *)malloc(sizeof(float)*mesh.nElements);										// array used to store RHS of the system of equations
	float *ConcentrationDist = (float *)malloc(sizeof(float)*mesh.nElements);			// array used to store the solution to the system of equations
	float *temp_ConcentrationDist = (float *)malloc(sizeof(float)*mesh.nElements);			// array used to store the solution to the system of equations

	// Initialize the concentration map with a linear gradient between the two boundaries
	for(int i = 0; i<mesh.numCellsY; i++){
		for(int j = 0; j<mesh.numCellsX; j++){
			ConcentrationDist[i*mesh.numCellsX + j] = (float)j/mesh.numCellsX*(opts.CRight - opts.CLeft) + opts.CLeft;
		}
	}

	// Zero the time

	myImg.gpuTime = 0;

	// Declare GPU arrays

	float *d_x_vec = NULL;
	float *d_temp_x_vec = NULL;
	
	float *d_Coeff = NULL;
	float *d_RHS = NULL;

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