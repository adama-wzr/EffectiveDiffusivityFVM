#ifndef _ASSC
#define _ASSC

#include <math.h>
#include <fstream>
#include <iostream>
#include <string>

#include <helper.cuh>
#include <Migration.cuh>
#include <datastructures.cpp>
#include <constants.cpp>
#include <solvers_2D.cuh>

/*

    Read input, assign defaults, print if necessary

*/

void printInputASSC(options *opts, ASSCopts *oASSC, meshInfo *mesh)
{
    /*
        printInputASSC Function:
        Inputs:
            - pointer to options struct
            - pointer to ASSC struct
        Outputs:
            - None.
        
        Function will print user entered options and
        defaults that have been stored in the data structures
        relevant to this simulation type.
    */

    // Print Model Options

    printf("--------------------------------------------\n\n");
    printf("            ASSC Model Inputs:              \n\n");
    printf("--------------------------------------------\n\n");

    printf("Input image name: %s\n", opts->inputFilename);

    printf("POI: %1.0d, Grayscale Threshhold = %3.0d\n", oASSC->POI, oASSC->POI_TH);

    printf("POI DC: %1.3e m^2/s\n", oASSC->POI_DC);

    printf("Initial concentration: %1.3e [mol/m^3]\n", oASSC->C0);

    printf("Pixel Resolution: %1.3e\n", oASSC->pixelRes);

    printf("Current Density: %1.3e A/m^2\n", oASSC->currentDensity);

    if (oASSC->C_or_D)
        printf("Simulating Charge Step\n");
    else
        printf("Simulating Discharge Step\n");

    printf("Start: %5.1lf (s), step: %4.1lf (s), final: %5.1lf (s)\n",
        oASSC->startTime, oASSC->stepTime, oASSC->totalTime);

    printf("--------------------------------------------\n\n");
    printf("            Simulation Options:             \n\n");
    printf("--------------------------------------------\n\n");

    printf("Reaction Control Mode: %d ", oASSC->mode);
    if(oASSC->mode == 0)
    {
        printf("(constant)\n");
    }else if(oASSC->mode == 1)
    {
        printf("(Tortuosity Weighed)\n");
        oASSC->TauE = oASSC->TauE;
        oASSC->TauLi = oASSC->TauLi;
        printf("TauLi = %1.3e, TauE = %1.3e\n", oASSC->TauLi, oASSC->TauE);
        oASSC->TauMax = (oASSC->TauE > oASSC->TauLi) ? oASSC->TauE : oASSC->TauLi;
    }

    // mesh amplificaiton

    printf("Mesh Refine X = %d\n", opts->MeshIncreaseX);
    printf("Mesh Refine Y = %d\n", opts->MeshIncreaseY);
    if (opts->nD == 3)
    {
        printf("Mesh Refine Z = %d\n", opts->MeshIncreaseZ);
    }

    // number of cells

    printf("nElements: %ld\n", mesh->nElements);
    printf("AM VF %1.3e\n", oASSC->AM_VF);
    printf("AM SA: %1.3e m^2\n", mesh->SA);
    printf("AM SSA: %1.3e m^-1\n", mesh->SSA);

    // Acceleration

    printf("Computation Mode:\n");

    if (opts->useGPU == 1)
    {
        printf("Using %d GPU(s)\n", opts->nGPU);
    }
    else
    {
        printf("Number of Threads = %d\n", opts->nThreads);
    }

    // convergence

    printf("Max. Iterations: %ld\n", opts->MAX_ITER);
    printf("Convergence: %1.3e\n", opts->ConvergeCriteria);
    printf("Time Step: %1.3e [s]\n", mesh->dt);


    // BC
    if(oASSC->PB)
        printf("BC = periodic\n");
    else
        printf("BC = no flux (sides)\n");


    return;
}

void readInputASSC(char *FileName, ASSCopts *oASSC)
{

    /*
        readInputASSC Function:
        Inputs:
            - FileName: pointer to where the input file name is stored.
            - ASSCopts: data structure options for ASSC simulation
        Outputs: None

        Function reads the input file for ASSC.
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
    oASSC->POI = 1;
    oASSC->POI_TH = 150;
    oASSC->POI_DC = 1e-10;  // m^2/s
    oASSC->C_or_D = 0;
    oASSC->printMAP = 0;
    oASSC->startTime = 0;
    oASSC->D0 = 1;
    oASSC->CMax = 1e15;
    oASSC->C0 = 1e4;    // mol/m^3
    oASSC->mode = 0;    // constant reaction rate

    oASSC->TauE = 1;
    oASSC->TauLi = 1;

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
            oASSC->POI = (int)tempD;
        }
        else if (strcmp(tempC, "POI_TH:") == 0)
        {
            oASSC->POI_TH = (int)tempD;
        }
        else if (strcmp(tempC, "POI_DC:") == 0)
        {
            oASSC->POI_DC = tempD;
        }
        else if (strcmp(tempC, "current_density:") == 0)
        {
            oASSC->currentDensity = tempD;
        }
        else if (strcmp(tempC, "startTime:") == 0)
        {
            oASSC->startTime = tempD;
        }
        else if (strcmp(tempC, "stepSize:") == 0)
        {
            oASSC->stepTime = tempD;
        }
        else if (strcmp(tempC, "totalTime:") == 0)
        {
            oASSC->totalTime = tempD;
        }
        else if (strcmp(tempC, "pixelResolution:") == 0)
        {
            oASSC->pixelRes = tempD;
        }
        else if (strcmp(tempC, "C_or_D:") == 0)
        {
            oASSC->C_or_D = (int)tempD;
        }
        else if (strcmp(tempC, "printMaps:") == 0)
        {
            oASSC->printMAP = (int)tempD;
        }
        else if (strcmp(tempC, "CMax:") == 0)
        {
            oASSC->CMax = tempD;
        }
        else if (strcmp(tempC, "Dprime:") == 0)
        {
            oASSC->D0 = tempD;
        }
        else if(strcmp(tempC, "C0:") == 0)
        {
            oASSC->C0 = tempD;
        }
        else if(strcmp(tempC, "PB:") == 0)
        {
            oASSC->PB = (int)tempD;
        }
        else if(strcmp(tempC, "Mode:") == 0)
        {
            oASSC->mode = (int)tempD;
        }
        else if(strcmp(tempC, "TauE:") == 0)
        {
            oASSC->TauE = tempD;
        }
        else if(strcmp(tempC, "TauLi:") == 0)
        {
            oASSC->TauLi = tempD;
        }
    }
    return;
}

void saveCyt_ASSC(meshInfo *mesh, double *Concentration, int step)
{
    /*
        Function saveCyt_ASSC:
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
    long int count = 0;

    for (int row = 0; row < mesh->numCellsY; row++)
    {
        double avgC = 0;
        count = 0;
        for (int col = 0; col < mesh->numCellsX; col++)
        {
            if (Concentration[row * mesh->numCellsX + col] == 0)
                continue;
            count++;
            avgC += Concentration[row * mesh->numCellsX + col];
        }
        if (count != 0)
            avgC = (double)avgC / count;
        else
            avgC = 0;
        fprintf(OUT, "%d,%1.3e\n", row, avgC);
    }

    fclose(OUT);

    return;
}

void printCandF_ASSC(options *opts, ASSCopts *oASSC, meshInfo *mesh, double *DC, double *C)
{
    /*
        Function printCandF_ASSC:
        Inputs:
            - pointer to options struct
            - pointer to oASSC struct
            - pointer to mesh struct
            - pointer to diffusion coefficients
            - pointer to Concentration
        Outputs:
            - None.

        Based on simulation data, the concentration and flux distributions will
        be printed. FLUX NOT IMPLEMENTED YET!
    */
    if (opts->verbose)
        printf("Printing Concentration map (fluxes not available yet)\n");

    // Open File
    FILE *MAP = fopen("sampleMaps.csv", "w+");

    fprintf(MAP, "x,y,C\n");

    for (int row = 0; row < mesh->numCellsY; row++)
    {
        for (int col = 0; col < mesh->numCellsX; col++)
        {
            // temporary storage
            int index = row * mesh->numCellsX + col;

            // If pore, skip

            // if (DC[index] == 0)
            // {
            //     fprintf(MAP, "%d,%d,%1.3e,%1.3e,%1.3e\n", col, row, 0.0f);
            //     continue;
            // }

            // print
            fprintf(MAP, "%d,%d,%1.3e\n", col, row, C[index]);

        } // endfor
    }

    // close file

    fclose(MAP);

    return;
}


/*

    Other auxiliary functions:

*/

double mode1_penalty_ASSC2D(ASSCopts *oASSC, meshInfo *mesh, double dcc, double dse)
{
    /*
        Function mode1_penalty_ASSC2D:
        Inputs:
            - pointer to ASSC options struct
            - pointer to mesh struct
            - distance (in pixels) from current collector
            - distance (in pixels) from solid electrolyte
        Outputs:
            - directly outputs the weighting factor.
    */
    
    double w = 0;

    w = 1 - pow((oASSC->TauE*dcc - oASSC->TauLi*dse),2) / 
                pow((oASSC->TauMax*mesh->numCellsY),2);

    return w;
}

void fixC_ASSC2D(meshInfo *mesh, double *DC, double *Conc)
{
    /*
        Function fixC_ASSC2D:
        Inputs:
            - pointer to struct mesh info
            - pointer to diffusion coefficient array
            - pointer to concentration array
        Outputs:
            - none
        
        For some small particles (a single pixel), the fluxes are too large
        compared to the amount of Li available. This can generate issues in
        the first iteration. This function here will regularize the Li 
        concentration after the first iteration to make sure these particles
        are just below the threshold for depletion while not having a negative
        concentration.
    */

    for(int i = 0; i < mesh->nElements; i++)
    {
        if (DC[i] == 0)
            continue;
        
        if (Conc[i] < 0)
        {
            Conc[i] = 100;
        }
    }

    return;
}

/*

    Discretization and Setup

*/

void ASSC2D_subDomainFF(meshInfo *mesh, ASSCopts *oASSC, char *simData, char *subDomain)
{
    /*
        ASSC2D_subDomainFF:
        Inputs:
            - pointer to meshInfo
            - pointer to ASSC options
            - pointer to simData
        Outputs:
            - none
        
        Function will use a flood-fill approach to characterize the number of
        independent active material (AM) subdomains. The information is stored
        in the subDomain array.
    */

    // make sure all entries in subDomain are -1

    for(int i = 0; i < mesh->nElements; i++)
    {
        subDomain[i] = -1;
    }

    // at the first point we find AM, set the counter to 1 and start the FF

    bool scan = true;
    
    int lastIdxChecked = 0;
    int nDomains = 0;

    int row, col;

    std::set<coordPair> cList;

    while (scan)
    {
        // find any solids that haven't been assigned yet
        for(int i = lastIdxChecked; i < mesh->nElements; i++)
        {
            if (simData[i] == oASSC->POI && subDomain[i] == -1)
            {
                lastIdxChecked = i;

                // open lists and assign starting point

                row = lastIdxChecked / mesh->numCellsX;
                col = lastIdxChecked - mesh->numCellsX * row;

                cList.insert(std::pair(col, row));

                // increase subdomain number
                nDomains++;
                subDomain[i] = nDomains;

                break;
            }
        }

        if (cList.empty())
        {
            scan = false;
        }

        

        while (!cList.empty())
        {
            // pop first item on the list
            coordPair pop = *cList.begin();

            // remove the item we just popped
            cList.erase(cList.begin());

            // read coordinates
            col = pop.first;
            row = pop.second;

            /*
                We need to check North, South, East, and West for more fluid:

                North = col + 0, row - 1
                South = col + 0, row + 1
                East  = col + 1, row + 0
                West  = col - 1, row + 0

                Note that diagonals are not considered a connection.
                If the user asks for periodic BCs, they are accounted for.
            */

            int tempRow, tempCol;
            long int tempIdx;

            // North

            tempCol = col;

            if (row > 0)
            {
                tempRow = row - 1;
                tempIdx = tempRow * mesh->numCellsX + tempCol;
                if (subDomain[tempIdx] == -1 && simData[tempIdx] == oASSC->POI)
                {
                    subDomain[tempIdx] = nDomains;
                    cList.insert(std::pair(tempCol, tempRow));
                }
            }

            // South

            if (row < mesh->numCellsY - 1)
            {
                tempRow = row + 1;
                tempIdx = tempRow * mesh->numCellsX + tempCol;
                if (subDomain[tempIdx] == -1 && simData[tempIdx] == oASSC->POI)
                {
                    subDomain[tempIdx] = nDomains;
                    cList.insert(std::pair(tempCol, tempRow));
                }
            }

            // East

            tempRow = row;

            if (col < mesh->numCellsX - 1)
            {
                tempCol = col + 1;
                tempIdx = tempRow * mesh->numCellsX + tempCol;
                if (subDomain[tempIdx] == -1 && simData[tempIdx] == oASSC->POI)
                {
                    subDomain[tempIdx] = nDomains;
                    cList.insert(std::pair(tempCol, tempRow));
                }
            }
            else if(col == mesh->numCellsX - 1 && oASSC->PB)
            {
                tempCol = 0;
                tempIdx = tempRow * mesh->numCellsX + tempCol;
                if (subDomain[tempIdx] == -1 && simData[tempIdx] == oASSC->POI)
                {
                    subDomain[tempIdx] = nDomains;
                    cList.insert(std::pair(tempCol, tempRow));
                }
            }

            // West

            if (col > 0)
            {
                tempCol = col - 1;
                tempIdx = tempRow * mesh->numCellsX + tempCol;
                if (subDomain[tempIdx] == -1 && simData[tempIdx] == oASSC->POI)
                {
                    subDomain[tempIdx] = nDomains;
                    cList.insert(std::pair(tempCol, tempRow));
                }
            }
            else if(col == 0 && oASSC->PB)
            {
                tempCol = mesh->numCellsX - 1;
                tempIdx = tempRow * mesh->numCellsX + tempCol;
                if (subDomain[tempIdx] == -1 && simData[tempIdx] == oASSC->POI)
                {
                    subDomain[tempIdx] = nDomains;
                    cList.insert(std::pair(tempCol, tempRow));
                }
            }
        } //end while
    }

    oASSC->nSubDomains = nDomains;

    return;
}


void activeSA_2D_ASSC(meshInfo *mesh, ASSCopts *oASSC, char *simData)
{
    /*
        activeSA_2D_ASSC:
        Inputs:
            - pointer to mesh info
            - pointer to ASSC opts
            - pointer to simData array
        Outputs:
            - none.
        
        Function will calculate the active surface area between AM and SE
        particles.
    */
    double SA = 0;
    int row, col;
    for(int i = 0; i < mesh->nElements; i++)
    {
        if(simData[i] != oASSC->POI)
            continue;
        row = i / mesh->numCellsX;
        col = i - row * mesh->numCellsX;

        // check north and south
        
        if (row != 0)
        {
            // check north
            if (simData[i - mesh->numCellsX] == 1)
            {
                SA += 1;
            }
        }

        if(row != mesh->numCellsY - 1)
        {
            // check South
            if (simData[i + mesh->numCellsX] == 1)
            {
                SA += 1;
            }
        }

        // check east and west

        if (col == 0 && oASSC->PB)
        {
            // periodic west
            int tempCol = mesh->numCellsX - 1;
            if(simData[row * mesh->numCellsX + tempCol] == 1)
            {
                SA += 1;
            }
        }
        
        if(col != 0)
        {
            // west
            if (simData[i - 1] == 1)
            {
                SA += 1;
            }
        }

        if (col == mesh->numCellsX - 1 && oASSC->PB)
        {
            // periodic east
            int tempCol = 0;
            if(simData[row*mesh->numCellsX + tempCol] == 1)
            {
                SA += 1;
            }
        }

        if (col != mesh->numCellsX - 1)
        {
            // east
            if(simData[i + 1] == 1)
            {
                SA += 1;
            }
        }
    }// end for

    // calculate SA and SSA
    mesh->numFaces = (long int) SA;
    mesh->SA = SA * mesh->dx;                           // number of faces times face area
    mesh->SSA = (double) mesh->SA / mesh->nElements;    // SA divided by volume

    return;
}

void ASSC_AM_VF(meshInfo *mesh, ASSCopts *oASSC, char *simData)
{
    /*
        ASSC_AM_VF Function:
        Inputs:
            - pointer to meshInfo struct
            - pointer to ASSC options struct
            - pointer to simData
        Outputs:
            - AM_VF
        
        Function will calculate the AM volume fraction for this simulation.
    */

    long int count = 0;

    for(int i = 0; i < mesh->nElements; i++)
    {
        if (simData[i] == oASSC->POI)
            count++;
    }

    oASSC->AM_VF = (double)count/mesh->nElements;

    return;
}

void ASSC_DC(meshInfo *mesh, ASSCopts *oASSC, char *simData, double *DC)
{
    /*
        ASSC_DC Function:
        Inputs:
            - pointer to meshInfo struct
            - pointer to ASSCopts struct
            - pointer to simData array
            - pointer to diffusion coefficient array
        Outputs:
            - none
        
        Function will binarize depending on the POI for the ASSC simulation.
        Only the DC array is modified.
    */

    for(int i = 0; i < mesh->nElements; i++)
    {
        if(simData[i] == oASSC->POI)
        {
            DC[i] = oASSC->POI_DC;
        }
        else
            DC[i] = 0;
    }

    return;
}

void SetBC_ASSC(options *opts, meshInfo *mesh, ASSCopts *oASSC, char *simData, int *BC, double *BC_value)
{
    /*
        Function SetBC_ASSC:
        Inputs:
            - pointer to opts
            - pointer to mesh
            - pointer to ASSCopts (submodel opts)
            - pointer to char simData
            - pointer to BC array
            - pointer to BC value (array)
        Outputs:
            - none
        
        BC Flags:
        0 : No boundary
        1 : Dirichlet
        2 : Neumann
        3 : Robin
        4 : Mixed
        5 : Periodic

        Note: 1, 3, and 4 are not used in this model.
        Function will populate BC_array and BC_value with the appropriate submodel boundary conditions
        and values for the BC's (if applicable).
    */

    // Set some variables to help
    int nCols, nRows;
    nCols = mesh->numCellsX + 2;
    nRows = mesh->numCellsY + 2;

    // Get applied current density

    double Area = mesh->numCellsX * oASSC->pixelRes;
    // double Area = PI * pow(0.002, 2);
    // double Area = 0.002;

    double appliedCurrent = oASSC->currentDensity * Area;

    /*
        Different Methods to calculate flux:
        
        1: same flux every active surface
        2: distance weighted
        3: ? 
    */

    // Currently only (1) is coded

    double flux;

    flux = mesh->SA/(mesh->dx * mesh->dy);

    oASSC->faceFlux = appliedCurrent / (opts->charge * FARADAY * mesh->numFaces);

    // search for boundaries

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
        BC[i * nCols + right] = 5;          // Periodic
        // left side
        BC[i * nCols + left] = 5;           // Periodic
    }

    // top and bottom boundaries

    for (int j = 0; j < nCols; j++)
    {
        // top
        BC[top * nCols + j] = 2;
        BC_value[top * nCols + j] = 0;
        // bottom (have to check it is not AM)

        BC[bottom * nCols + j] = 2;
        // BC_value[bottom * nCols + j] = -appliedCurrent;
        BC_value[bottom * nCols + j] = 0;   // test
    }

    return;
}

void subAvgC_ASSC2D(meshInfo    *mesh,
                    ASSCopts    *oASSC,
                    double      *C0,
                    char        *subDomain,
                    int         *subSize,
                    double      *subC)
{
    /*
        Function subAvgC_ASSC2D:
        Inputs:
            - pointer to the mesh struct
            - pointer to the oASSC struct
            - pointer to concentration array
            - pointer to subDomain classification array
            - pointer to sub-domain size array
            - pointer to sub-domain average concentration array
        Outputs:
            - none

        Function will calculate the size of each subdomain and calculate
        the average concentration in each subdomain.
    */

    for(int index = 0; index < mesh->nElements; index++)
    {
        if (subDomain[index] == -1)
            continue;

        // Now we know this is some subdomain

        int subIdx = subDomain[index] - 1;

        subSize[subIdx]++;
        subC[subIdx] += C0[index];
    }

    // Average and print
    for(int index = 0; index < oASSC->nSubDomains; index++)
    {
        subC[index] = (double)subC[index] / subSize[index];
    }

    return;
}

void disc2D_ASSC(options     *opts,
                meshInfo    *mesh,
                ASSCopts    *oASSC,
                double      *DC,
                double      *Coeff,
                double      *RHS,
                double      *C0,
                char        *simData,
                char        *subDomain,
                double      *subAvgC)
{
    /*
        Function disc2D_ASSC:
        Inputs:
            - pointer to options struct
            - pointer to mesh struct
            - pointer to ASSC opts
            - pointer to DC
            - pointer to Coefficient Matrix
            - pointer to RHS
            - pointer to concentration dist. at last time-step
            - pointer to simData (phase-spec) array
            - pointer to subDomain array
            - pointer to subAvgC
        Outputs:
            - None.
        
        Function will create a 2D + 1D discretization of the given system based on
        central differencing for the space dependent component and Crank-Nicolson
        method for implicit time stepping.
    */

    // Set necessary variables

    int nCols;
    nCols = mesh->numCellsX;

    double dx, dy, dt;
    dx = mesh->dx;
    dy = mesh->dy;
    dt = mesh->dt;

    int row, col;
    long int BC_index;
    double dw, de, ds, dn;

    int tempE, tempW;

    int sdIdx;

    // main loop

    for(int index = 0; index < mesh->nElements; index++)
    {
        // If DC = 0, do not solve
        if(DC[index] == 0)
        {
            Coeff[index*5 + 0] = 1;
            RHS[index] = 0;
            continue;
        }

        // make sure Coeff and RHS are 0
        RHS[index] = 0;
        for (int k = 0; k < 5; k++)
        {
            Coeff[index * 5 + k] = 0;
        }

        // Deal with current conditions

        row = index / nCols;
        col = index - row * nCols;

        // get temp indexes in case of periodic BC

        if (col == 0)
        {
            tempW = nCols - 1;
            tempE = col + 1;
        }
        else if (col == nCols - 1)
        {
            tempW = col - 1;
            tempE = 0;
        }
        else
        {
            tempW = col - 1;
            tempE = col + 1;
        }

        /*
            Indexing for coeff marix:

            0 : P       i
            1 : W       i - 1
            2 : E       i + 1
            3 : S       i + nCols
            4 : N       i - nCols
        */

        // RHS

        RHS[index] += 2.0 * (dx * dy)/dt * C0[index];

        // West

        if(simData[row * nCols + tempW] == 1)
        {
            if (oASSC->mode == 0)
            {
                RHS[index] += -oASSC->faceFlux;
            }
            else if(oASSC->mode == 1)
            {
                RHS[index] += -2 * oASSC->faceFlux *
                         mode1_penalty_ASSC2D(oASSC, mesh, (double)row + 1, (double)(mesh->numCellsY - row));
            }
        }
        else
        {
            dw = WeightedHarmonicMean(dx/2, dx/2, DC[index], DC[row * nCols + tempW]);
            Coeff[index*5 + 1] = -dw * dx/dy;
            Coeff[index*5 + 0] += dw * dx/dy;
            RHS[index] -= Coeff[index*5 + 1] * C0[row * nCols + tempW];
        }

        // East

        if(simData[row * nCols + tempE] == 1)
        {
            if (oASSC->mode == 0)
            {
                RHS[index] += -oASSC->faceFlux;
            }
            else if(oASSC->mode == 1)
            {
                RHS[index] += -2*oASSC->faceFlux *
                         mode1_penalty_ASSC2D(oASSC, mesh, (double)row + 1, (double)(mesh->numCellsY - row));
            }
        }
        else
        {
            de = WeightedHarmonicMean(dx/2, dx/2, DC[index], DC[row * nCols + tempE]);
            Coeff[index*5 + 2] = -de * dx/dy;
            Coeff[index*5 + 0] += de * dx/dy;
            RHS[index] -= Coeff[index*5 + 2] * C0[row * nCols + tempE];
        }

        // South

        if (row != mesh->numCellsY - 1)
        {
            if(simData[(row + 1) * nCols + col] == 1)
            {
                if (oASSC->mode == 0)
                {
                    RHS[index] += -oASSC->faceFlux;
                }
                else if(oASSC->mode == 1)
                {
                    RHS[index] += -2*oASSC->faceFlux *
                            mode1_penalty_ASSC2D(oASSC, mesh, (double)row + 1, (double)(mesh->numCellsY - row));
                }
            }
            else
            {
                ds = WeightedHarmonicMean(dy/2, dy/2, DC[index], DC[(row + 1) * nCols + col]);
                Coeff[index * 5 + 3] = -ds * dy / dx;
                Coeff[index * 5 + 0] += ds * dy/dx;
                RHS[index] -= Coeff[index*5 + 3] * C0[(row + 1) * nCols + col]; 
            }
        }

        // North

        if(row != 0)
        {
            if(simData[(row - 1) * nCols + col] == 1)
            {
                if (oASSC->mode == 0)
                {
                    RHS[index] += -oASSC->faceFlux;
                }
                else if(oASSC->mode == 1)
                {
                    RHS[index] += -2*oASSC->faceFlux *
                            mode1_penalty_ASSC2D(oASSC, mesh, (double)row + 1, (double)(mesh->numCellsY - row));
                }
            }
            else
            {
                dn = WeightedHarmonicMean(dy/2, dy/2, DC[(row - 1) * nCols + col], DC[index]);
                Coeff[index * 5 + 4] = -dn * dy/dx;
                Coeff[index * 5 + 0] += dn * dy/dx;
                RHS[index] -= Coeff[index * 5 + 4] * C0[(row - 1) * nCols + col];
            }
        }

        // P contribution from last time step

        RHS[index] += -Coeff[index*5 + 0] * C0[index];
        Coeff[index*5 + 0] += 2.0 * dx * dy / dt;

        //end
    }

    return;
}


int RHS_Up2D_ASSC(meshInfo   *mesh,
                ASSCopts    *oASSC,
                double      *DC,
                double      *CoeffMatrix,
                double      *RHS,
                double      *C0,
                char        *simData,
                char        *subDomain,
                double      *subAvgC)
{

    /*
        Function RHS_Up2D_ASSC:
        Inputs:
            - pointer to mesh struct
            - pointer to oASSC options
            - pointer to diffusion coefficient array.
            - pointer to CoeffMatrix array
            - pointer to RHS array
            - pointer to concentration values array from previous time step
            - pointer to simData array.
            - pointer to subDomain labels
            - pointer to subDomain average concentration
        Outputs:
            - None.
        
        Function will update the RHS matrix according to the values from previous time-step.
        Unless there is an update to BCs, then the Coeff Matrix does not see any changes.
    */

    // Set necessary variables

    int nCols;
    nCols = mesh->numCellsX;

    double dx, dy, dt;

    dx = mesh->dx;
    dy = mesh->dy;
    dt = mesh->dt;

    int row, col;
    long int BC_index;
    double ap;

    int tempE, tempW;

    for (long int i = 0; i < mesh->nElements; i++)
    {
        // dissolve index into rows and cols
        row = i / nCols;
        col = i - row * nCols;

        if(DC[i] == 0)
        {
            // non-participating media does not need an update
            continue;
        }

        // This means participating fluid and not a wall

        /*
            Indexing for coeff marix:

            0 : P       i
            1 : W       i - 1
            2 : E       i + 1
            3 : S       i + nCols
            4 : N       i - nCols
        */

        // Reset RHS
        RHS[i] = 0;
        ap = 0;

        // Contribution from last time step

        RHS[i] += 2.0 * (dx * dy)/dt * C0[i];

        // get a_p = sum(a_nb)

        for(int j = 1; j < 5; j++)
        {
            ap += -CoeffMatrix[i * 5 + j];
        }

        // get subdomain idx

        int subIdx = subDomain[i] - 1;
        double avgC = subAvgC[subIdx];

        // Check all directions for BCs

        // Get periodic BC's

        if (col == 0)
        {
            // periodic West
            tempW = mesh->numCellsX - 1;
            tempE = col + 1;
        } else if(col == mesh->numCellsX - 1)
        {
            // periodic East
            tempW = col - 1;
            tempE = 0;
        }
        else
        {
            // no boundaries
            tempW = col - 1;
            tempE = col + 1;
        }

        // West

        if(DC[row * nCols + tempW] != 0)
        {
            // contribution from the last time-step
            RHS[i] += -CoeffMatrix[i * 5  + 1] * C0[row * nCols + tempW];
        }
        else if(simData[row * nCols + tempW] == 1 && avgC > 300)
        {
            // contribution from BC flux
            if (oASSC->mode == 0)
            {
                RHS[i] += -oASSC->faceFlux;
            }
            else if(oASSC->mode == 1)
            {
                RHS[i] += -2*oASSC->faceFlux *
                        mode1_penalty_ASSC2D(oASSC, mesh, (double)row + 1, (double)(mesh->numCellsY - row));
            }
        }

        // East

        if(DC[row * nCols + tempE] != 0)
        {
            // contribution from last time-step
            RHS[i] += -CoeffMatrix[i * 5 + 2] * C0[row * nCols + tempE];
        }
        else if(simData[row * nCols + tempE] == 1 && avgC > 300)
        {
            // contribution from BC flux
            if (oASSC->mode == 0)
            {
                RHS[i] += -oASSC->faceFlux;
            }
            else if(oASSC->mode == 1)
            {
                RHS[i] += -2*oASSC->faceFlux *
                        mode1_penalty_ASSC2D(oASSC, mesh, (double)row + 1, (double)(mesh->numCellsY - row));
            }
        }

        // South

        if(row != mesh->numCellsY - 1)
        {
            if(simData[(row + 1) * nCols + col] == 1 && avgC > 300)
            {
                // contribution from BC flux
                if (oASSC->mode == 0)
                {
                    RHS[i] += -oASSC->faceFlux;
                }
                else if(oASSC->mode == 1)
                {
                    RHS[i] += -2*oASSC->faceFlux *
                            mode1_penalty_ASSC2D(oASSC, mesh, (double)row + 1, (double)(mesh->numCellsY - row));
                }
            }
            else if(DC[(row + 1) * nCols + col] != 0)
            {
                // prev. step contribution
                RHS[i] -= CoeffMatrix[i * 5 + 3] * C0[(row + 1) * nCols + col];
            }
        }

        // North

        if (row != 0)
        {
            if(simData[(row - 1) * nCols + col] == 1 && avgC > 300)
            {
                // contribution from BC flux
                if (oASSC->mode == 0)
                {
                    RHS[i] += -oASSC->faceFlux;
                }
                else if(oASSC->mode == 1)
                {
                    RHS[i] += -2*oASSC->faceFlux *
                            mode1_penalty_ASSC2D(oASSC, mesh, (double)row + 1, (double)(mesh->numCellsY - row));
                }
            }
            else if(DC[(row - 1) * nCols + col] != 0)
            {
                // prev. step contribution
                RHS[i] -= CoeffMatrix[i * 5 + 4] * C0[(row - 1) * nCols + col];
            }
        }

        // last contribution is ap

        RHS[i] += -ap * C0[i];
    }

    return 0;
}



// Test function below

void test_funct(void)
{
    printf("hello world\n");

    return;
}


#endif