#ifndef _TSSD
#define _TSSD

#include <math.h>
#include <fstream>
#include <iostream>
#include <string>

#include <helper.cuh>
#include <Migration.cuh>
#include <datastructures.cpp>
#include <constants.cpp>

/*

Handling user input in TSSD submodel:

*/

void printTSSD(options *opts, TSSDopts *oTSSD, Migration *mig)
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

    printf("Initial Concentration: %1.3e mol/m^3\n", oTSSD->C0);

    if (oTSSD->C_or_D)
        printf("Simulating Charge\n");
    else
        printf("Simulating Discharge Step\n");

    printf("Current Density: %1.3f A/m^2\n", oTSSD->current_density);

    if(oTSSD->useGITT)
    {
        printf("Reading GITT Results.\n");
        printf("GITT File Name: %s\n", oTSSD->GITT_Name);
    }

    if(oTSSD->useLinear)
    {
        printf("Using Linear DC-to-C correlation.\n");
    }

    if(oTSSD->useAnom)
    {
        printf("\n--------------------------------------------\n\n");
        printf("Using Anomalous Diffusion Model\n");
        printf("Cmax: %1.3e [mol/m3]\n", oTSSD->CMax);
        printf("D': %1.3e [m^2/s]\n", oTSSD->Dprime);
    }

    if(oTSSD->useMig)
    {
        printf("\n--------------------------------------------\n\n");
        printf("Migration Phenomena Considerations:\n");
        printf("Charge = %d\n", opts->charge);
        printf("Temp = %3.1f Kelvin\n", mig->T);
        printf("E-Field Gradient (x) =  %1.3e\n", mig->dE_dL[0]);
        printf("E-Field Gradient (y) =  %1.3e\n", mig->dE_dL[1]);
        printf("E-Field Gradient (z) =  %1.3e\n", mig->dE_dL[2]);
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
    oTSSD->useGITT = 0;
    oTSSD->useLinear = 0;
    oTSSD->useAnom = 0;
    oTSSD->useMig = 0;
    oTSSD->Dprime = 0;
    oTSSD->C0 = 1.65;

    oTSSD->GITT_Name = (char *)malloc(sizeof(char) * 1000);

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
        else if(strcmp(tempC, "useGITT:") == 0)
        {
            oTSSD->useGITT = (int)tempD;
        }
        else if (strcmp(tempC, "GITT_File:") == 0)
        {
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            strcpy(oTSSD->GITT_Name, tempFilenames);
        }
        else if(strcmp(tempC, "Linear:") == 0)
        {
            oTSSD->useLinear = (int)tempD;
        }
        else if(strcmp(tempC, "Anomalous:") == 0)
        {
            oTSSD->useAnom = (int)tempD;
        }
        else if (strcmp(tempC, "CMax:") == 0)
        {
            oTSSD->CMax = tempD;
        }
        else if (strcmp(tempC, "Dprime:") == 0)
        {
            oTSSD->Dprime = tempD;
        }
        else if(strcmp(tempC, "C0:") == 0)
        {
            oTSSD->C0 = tempD;
        }
        else if(strcmp(tempC, "Mig:") == 0)
        {
            oTSSD->useMig = (int)tempD;
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
        avgC = (double)avgC / count;
        fprintf(OUT, "%d,%1.3e\n", row, avgC);
    }

    fclose(OUT);

    return;
}

void printCandF(options *opts, TSSDopts *oTSSD, meshInfo *mesh, double *DC, double *C)
{
    /*
        Function printCandF:
        Inputs:
            - pointer to options struct
            - pointer to oTSSD struct
            - pointer to mesh struct
            - pointer to diffusion coefficients
            - pointer to Concentration
        Outputs:
            - None.
        
        Based on simulation data, the concentration and flux distributions will
        be printed.
    */

    // Open File
    FILE *MAP = fopen("sampleMaps.csv", "w+");

    fprintf(MAP, "x,y,C,Jx,Jy\n");

    for(int row = 0; row < mesh->numCellsY; row++)
    {
        for(int col = 0; col < mesh->numCellsX; col++)
        {
            // temporary storage
            int index = row * mesh->numCellsX + col;
            double Jx, Jy;
            double J1, J2;  

            // If pore, skip

            if (DC[index] == 0)
            {
                fprintf(MAP, "%d,%d,%1.3e,%1.3e,%1.3e\n", col, row, 0.0f , 0.0f, 0.0f);
                continue;
            }

            // calc Jx first
            J1 = 0;
            J2 = 0;
            if (col != 0)
            {
                if (DC[index - 1] != 0)
                {
                    // West exists
                    J1 = WeightedHarmonicMean(mesh->dx / 2, mesh->dx / 2, DC[index], DC[index - 1]) *
                         (C[index - 1] - C[index])/mesh->dx;
                }
                else
                {
                    J1 = 0;
                }
            }

            if (col != mesh->numCellsX - 1)
            {
                if (DC[index + 1] != 0)
                {
                    // West exists
                    J2 = WeightedHarmonicMean(mesh->dx / 2, mesh->dx / 2, DC[index], DC[index + 1]) *
                         (C[index] - C[index + 1])/mesh->dx;
                }
                else
                {
                    J2 = 0;
                }
            }
            
            // get Jx
            if(col == 0)
            {
                Jx = J2;
            }
            else if(col == mesh->numCellsX - 1)
            {
                Jx = J1;
            }
            else
            {
                Jx = (J1 + J2)/2;
            }

            // Get Jy
            J1 = 0;
            J2 = 0;

            if(row !=0)
            {
                if(DC[index - mesh->numCellsX] != 0)
                {
                    // North Exists
                    J1 = WeightedHarmonicMean(mesh->dy/2, mesh->dy/2, DC[index], DC[index - mesh->numCellsX]) * 
                        (C[index] - C[index - mesh->numCellsX])/mesh->dy;
                }
                else
                {
                    J1 = 0;
                }
            }

            if (row != mesh->numCellsY - 1)
            {
                if(DC[index + mesh->numCellsX] != 0)
                {
                    // North Exists
                    J2 = WeightedHarmonicMean(mesh->dy/2, mesh->dy/2, DC[index], DC[index + mesh->numCellsX]) * 
                        (C[index + mesh->numCellsX] - C[index])/mesh->dy;
                }
                else
                {
                    J2 = 0;
                }
            }

            if(row == 0)
            {
                Jy = J2;
            }
            else if(row == mesh->numCellsY - 1)
            {
                if ( mesh->SA == 0)
                {
                    J2 = 0;
                }
                else
                {
                    J2 = mesh->dt * oTSSD->current_density / (mesh->SA * opts->charge * FARADAY);
                }
                Jy = (J2 + J1)/2;
            }
            else
            {
                Jy = (J2 + J1)/2;
            }

            // print
            fprintf(MAP, "%d,%d,%1.3e,%1.3e,%1.3e\n", col, row, C[index] , Jx, Jy);


        } //endfor
    }

    // close file

    fclose(MAP);

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

    // volume of the sample is hard-coded, consider making this an input
    double volume = 3.1415 * 100e-06 * pow(0.004,2)/4.0;
    // convert current density to current
    oTSSD->current_density = oTSSD->current_density * 3.1415 * pow(0.004,2)/4.0;
    // flux units = mol m^-2 s^-1
    flux = oTSSD->current_density / (mesh->SSA/(pow(oTSSD->pixelRes, 3)) * volume * opts->charge * FARADAY);
    flux = 2*2.8599e-05;

    printf("SSA: %1.3e m^-1, Volume  = %1.3e m^3, current = %1.3e A\n", mesh->SSA/pow(oTSSD->pixelRes, 3), volume, oTSSD->current_density);
    printf("SA: %1.3e m^2, Flux = %1.3e [mol/m^2-s]\n", mesh->SSA/(pow(oTSSD->pixelRes, 3)) * volume, flux);

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
        {   // charge
            BC[bottom * nCols + j] = 2;
            BC_Value[bottom * nCols + j] = -flux;
        }
        else
        {   // discharge
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

/*

    GITT Related Material:

*/

int GITT_Interval(TSSDopts *oTSSD, double *SOC, double *GITT_D, int *nData)
{
    /*
        Function GITT_Interval:
        Inputs:
            - options TSSD, for file name
            - SOC is pointer to pre-allocated array that will hold
                the SOC (or DoD) steps
            - GITT_D will hold the diffusion according to GITT, array
                is pre-allocated
            - pointer to nData: gives amount of data pre-allocated (for
                error checking), and then stores the new amount after reading.
        Outputs:
            - none.
        
        Function will read the GITT results, and store them for usage later on.
    */

    // open file, start reading

    FILE *target_data;

    target_data = fopen(oTSSD->GITT_Name, "r");

    // check if file exists

    if (target_data == NULL)
    {
        fprintf(stderr, "Error reading file. Exiting program.\n");
        return 1;
    }

    // read header

    char header1[20];
    char header2[20];

    fscanf(target_data, "%s,%s,", &header1[0], &header2[0]);

    printf("Debug Header = %s %s\n", header1, header2);

    size_t count = 0;

    while (fscanf(target_data, "%lf,%lf", &SOC[count], &GITT_D[count]) == 2)
    {
        count++;
        if (count > *nData)
        {
            printf("Not Enough Space Allocated. Exiting...\n");
            return 1;
        }
    }

    // update the number of data
    *nData = (int) count;

    // close the file
    fclose(target_data);

    return 0;
}

void SetDC_Linear(options *opts, TSSDopts *oTSSD, meshInfo *mesh, double *DC, char *simData, double *C)
{
    /*
        Function SetDC_Linear:
        Inputs:
            - pointer to options struct
            - pointer to TSSD options
            - pointer to mesh struct
            - pointer to diffusion coefficients
            - pointer to simData (phase labels)
            - pointer to concentration array
        Outputs:
            - none
        
        Diffusion coefficients are set based on the current concentration,
        therefore diffusion coefficient is dependent on concentration
        ONLY FOR THE POI.
    */

    // hardcoded values (empirically derived by neutron + GITT)
    
    double a = 2.1727e-13;
    double b = -2.9780e-13;

    // iterate over the whole domain

    for(int row = 0; row < mesh->numCellsY; row++)
    {
        for(int col = 0; col < mesh->numCellsX; col++)
        {
            int localPhase = simData[row * mesh->numCellsX + col];
            if(localPhase != oTSSD->POI)
            {
                DC[row * mesh->numCellsX + col] = opts->DC[localPhase];
            }
            else
            {
                DC[row * mesh->numCellsX + col] = a * C[row * mesh->numCellsX + col] + b;
            }
        }
    }

    return;
}

void SetDC_GITT(options *opts, TSSDopts *oTSSD, meshInfo *mesh, double *DC, char *simData, double POI_DC)
{
    /*
    
        Function SetDC_GITT:
        Input:
            - pointer to options struct, with general user input options
            - oTSSD is a pointer to the data struct holding TSSD input data
            - pointer to the mesh array
            - pointer to diffusion coefficient array
            - pointer to simData
            - value of diffusion coefficient to be used for POI
        Output:
            - none
        
        Function will modify the DC-array to hold the proper DC according 
        to GITT for the POI.
    */

    // iterate over simData

    for(int row = 0; row < mesh->numCellsY; row++)
    {
        for(int col = 0; col < mesh->numCellsX; col++)
        {
            int localPhase = simData[row * mesh->numCellsX + col];
            if(localPhase != oTSSD->POI)
            {
                DC[row * mesh->numCellsX + col] = opts->DC[localPhase];
            }
            else
            {
                DC[row * mesh->numCellsX + col] = POI_DC;
            }
        }
    }

    return;
}

int setDC_AnomDiff(TSSDopts *oTSSD, meshInfo *mesh, double *DC, double *C, char *simData)
{
    /*
        Function setDC_AnomDiff:
        Inputs:
            - pointer to TSSD opts
            - pointer to mesh info
            - pointer to DC array
            - pointer to Concentration array
            - pointer to simData array (phase info)
        Outputs:
            - None
        
        Function will use user entered information along with the concentration array
        to provide the diffusion coefficient of the POI according to the theory of
        anomalous diffusion:

        D = D'(Cmax + C)/(Cmax - C)

        where D' and Cmax are entered by the user, C is using the calculated concentration.

        The function returns false if Cmax - C ~ 0
    */

    for (int i = 0; i < mesh->nElements; i++)
    {
        // get local phase
        int localPhase = simData[i];
        
        // check if not active material, increment
        if(localPhase != oTSSD->POI)
            continue;
        // check for NaN potential
        if (oTSSD->CMax <= C[i])
            return 1;

        DC[i] = oTSSD->Dprime*(C[i] + oTSSD->CMax)/(oTSSD->CMax - C[i]);
    }
    
    return 0;
}

#endif