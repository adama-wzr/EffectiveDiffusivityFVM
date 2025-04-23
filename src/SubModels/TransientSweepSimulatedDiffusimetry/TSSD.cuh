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
    printf("Pixel Resolution: %1.3e\n", oTSSD->pixelRes);
    for (int i = 0; i < opts->numDC; i++)
    {
        printf("Phase = %d\n", i + 1);
        printf("Threshold (Upper Bound) = %d\n", opts->DC_TH[i]);
        if (i == oTSSD->POI - 1)
            printf("DC[%d] = ???\n", i + 1);
        else
        {
            printf("DC[%d] = %1.3e m^2/s\n", i + 1, opts->DC[i]);
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

    if (oTSSD->printMAP)
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

    if (oTSSD->C_or_D)
        printf("Simulating Charge\n");
    else
        printf("Simulating Discharge Step\n");

    printf("Current Density: %1.3f A\n", oTSSD->current_density);

    if (oTSSD->D0 == 1)
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
        else if (strcmp(tempC, "pixelResolution:") == 0)
        {
            oTSSD->pixelRes = tempD;
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

/*

    Output Handling Functions:

*/

void saveCyt(meshInfo *mesh, double *Concentration, int step)
{
    /*
        Function saveCyt:
        Inputs:
            - pointer to mesh struct
            - pointer to concentration array
            - interger step
        Outputs:
            - none

        The function will simply calculate the average concentration in each column,
        thus returning the average concentration as function of y at a given time step.

        The output folder is created if it doesn't exist, and the files names
        are indexed by the simulations save-step number.
    */

    // folder and file names
    char foldername[100];
    char filename[100];

    sprintf(foldername, "OutputCyt");
    sprintf(filename, "Cyt_%05d.csv", step);

    // check if folder exists
    if (!std::filesystem::is_directory(foldername) || !std::filesystem::exists(foldername))
    {
        // create folder
        std::filesystem::create_directory(foldername);
    }

    std::filesystem::path dir(foldername);
    std::filesystem::path file(filename);
    std::filesystem::path full_path = dir / file;

    // open file and save cmap

    FILE *OUT;

    OUT = fopen(full_path.generic_string().c_str(), "w");

    fprintf(OUT, "y,Cy\n");

    for (int row = 0; row < mesh->numCellsY; row++)
    {
        double avgC = 0;
        for (int col = 0; col < mesh->numCellsX; col++)
        {
            avgC += Concentration[row * mesh->numCellsX + col];
        }
        avgC = (double)avgC / mesh->numCellsX;
        fprintf(OUT, "%d,%1.3e\n", row, avgC);
    }

    fclose(OUT);

    return;
}

/*

    Boundary conditions for TSSD

*/

void SetBC_TSSD2D(options *opts, TSSDopts *oTSSD, meshInfo *mesh, int *BC, double *BC_Value)
{
    /*
        SetBC_TSSD2D:
        Inputs:
            - pointer to opts struct
            - pointer to TSSDopts struct
            - pointer to meshInfo struct
            - pointer to BC (flags)
            - pointer to BC_Value (BC values)
        Outputs:
            - None.

        The function takes user input into account to build the BC setup
        for the TSSD model simulation.

        BC Flags:
        0 : No boundary
        1 : Dirichlet
        2 : Neumann
        3 : Robin
        4 : Mixed

        Note: 1, 3, and 4 are not used in this model.
    */

    // Set some variables to help
    int nCols, nRows;
    nCols = mesh->numCellsX + 2;
    nRows = mesh->numCellsY + 2;

    double flux;

    flux = oTSSD->current_density / (mesh->SA * opts->charge * FARADAY);

    printf("SA: %1.3e m^2, Flux = %1.3e [units?]\n", mesh->SA, flux);

    int right, left, top, bottom;
    // set col values for right and left
    left = 0;
    right = nCols - 1;

    // set row values for top and bottom
    top = 0;
    bottom = nRows - 1;

    // right and left boundaries (Neumann)

    for (int i = 0; i < nRows; i++)
    {
        // right side
        BC[i * nCols + right] = 2; // Neumann
        BC_Value[i * nCols + right] = 0;

        // left side
        BC[i * nCols + left] = 2; // Neumann
        BC_Value[i * nCols + left] = 0;
    }

    // set Neumann boundaries

    for (int j = 0; j < nCols; j++)
    {
        // top
        BC[top * nCols + j] = 2;
        BC_Value[top * nCols + j] = 0;

        // bottom
        if (oTSSD->C_or_D == 0)
        {
            BC[bottom * nCols + j] = 2;
            BC_Value[bottom * nCols + j] = -flux;
        }
        else
        {
            BC[bottom * nCols + j] = 2;
            BC_Value[bottom * nCols + j] = flux;
        }
    }

    return;
}

/*

Flood Fill Setup:

*/

int FloodFill2D_Bot(meshInfo *mesh, int *BC, double *DC)
{
    /*
        FloodFill2D_Bot function:
        Inputs:
            - pointer to mesh struct
            - pointer to array with BC's
            - pointer to array with DC's
        Outputs:
            - None

        The function will search the domain, and will set all DC values that are too
        low to a Neumann BC with zero flux. Non-participating media will also be flagged
        accordingly. This function starts from the bottom boundary.
    */

    char *Domain = (char *)malloc(mesh->nElements * sizeof(char));

    // Initialize all the impermeable matter in the domain:

    for (long int index = 0; index < mesh->nElements; index++)
    {
        int row = index / mesh->numCellsX;
        int col = index - row * mesh->numCellsX;

        long int indexBC = (row + 1) * (mesh->numCellsX + 2) + (col + 1);
        if (DC[index] == 0)
        {
            Domain[index] = 0;
            BC[indexBC] = 2;
        }
        else
        {
            Domain[index] = -1;
        }
    }

    // Find permeable boundaries, add to open list

    std::set<coordPair> cList;

    int bot = mesh->numCellsY - 1;

    for (int col = 0; col < mesh->numCellsX; col++)
    {
        // set bot
        if (Domain[bot * mesh->numCellsX + col] == -1)
        {
            Domain[bot * mesh->numCellsX + col] = 0;
            cList.insert(std::pair(col, bot));
        }
    }

    // Search full domain

    while (!cList.empty())
    {
        // pop first item on the list
        coordPair pop = *cList.begin();

        // remove from open list
        cList.erase(cList.begin());

        // read coordinates

        int col = pop.first;
        int row = pop.second;

        /*
            We need to check North, South, East, and West for more fluid:

            North = col + 0, row - 1
            South = col + 0, row + 1
            East  = col + 1, row + 0
            West  = col - 1, row + 0

            Note that diagonals are not considered a connection.
            This code assumes no periodic boundary conditions (currently).
        */
        int tempRow, tempCol;
        long int tempIndex;

        // North

        tempCol = col;

        if (row > 0)
        {
            tempRow = row - 1;
            tempIndex = tempRow * mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::pair(tempCol, tempRow));
            }
        }

        // South

        tempCol = col;

        if (row < mesh->numCellsY - 1)
        {
            tempRow = row + 1;
            tempIndex = tempRow * mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::pair(tempCol, tempRow));
            }
        }

        // West

        tempRow = row;

        if (col > 0)
        {
            tempCol = col - 1;
            tempIndex = tempRow * mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::pair(tempCol, tempRow));
            }
        }

        // East

        tempRow = row;

        if (col < mesh->numCellsX - 1)
        {
            tempCol = col + 1;
            tempIndex = tempRow * mesh->numCellsX + tempCol;
            if (Domain[tempIndex] == -1)
            {
                Domain[tempIndex] = 0;
                cList.insert(std::pair(tempCol, tempRow));
            }
        }

        // end while
    }

    // Every flag that is still -1 means a non-participating media

    for (int index = 0; index < mesh->nElements; index++)
    {
        // Skip participating media
        if (Domain[index] != -1)
            continue;

        int row = index / mesh->numCellsX;
        int col = index - row * mesh->numCellsX;

        long int indexBC = (row + 1) * (mesh->numCellsX + 2) + (col + 1);

        // Set BC of non-participating media

        BC[indexBC] = -1;
    }

    // memory management
    free(Domain);

    return 0;
}

#endif