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
	unsigned char *target_data;
	float keff;
	bool PathFlag;
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

// define pair <double, pair<i,j>> for open list

typedef std::pair<float, std::pair<int,int> > OpenListInfo;


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

	// Now use the information gathered 




	return 0;
}