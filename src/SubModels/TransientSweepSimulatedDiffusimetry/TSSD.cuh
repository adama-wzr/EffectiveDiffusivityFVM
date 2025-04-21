#include <math.h>
#include <helper.cuh>

/*

Handling user input in TSSD submodel:

*/

void readInputTSSD(char *FileName, options *opts)
{

    /*
        readInputTSSD Function:
        Inputs:
            - FileName: pointer to where the input file name is stored.
            - struct options: pass a struct with the options.
        Outputs: None

        Function reads the input file and stores the options in the opts struct,
        specifically for TSSD subroutine simulation.
    */

    // initiate necessary variables for input reading
    std::string myText;

    char tempC[1000];
    double tempD;
    char tempFilenames[1000];
    std::ifstream InputFile(FileName);

    // initiate arrays for storing names of input/output files

    opts->inputFilename = (char *)malloc(1000 * sizeof(char));
    opts->outputFilename = (char *)malloc(1000 * sizeof(char));
    opts->CMapName = (char *)malloc(1000 * sizeof(char));
    opts->FMapName = (char *)malloc(1000 * sizeof(char));
    opts->StartMapName = (char *)malloc(1000 * sizeof(char));

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
    opts->Time = 0.0;
    opts->current = 0;
    opts->charge = 0;
    opts->StartTime = 0;

    opts->nThreads = 1;

    opts->useGPU = 0;
    opts->nGPU = 1;

    opts->SteadyStateFlag = 0;
    opts->tauSim = 0;
    opts->TF_Flag = 0;
    opts->StartMapFlag = 0;

    /*
    --------------------------------------------------------------------------------

    If anybody has a better idea of how to parse inputs please let me know.
    Eventually I'm hoping the GUI will replace a lot of this code.

    --------------------------------------------------------------------------------
    */

    while (std::getline(InputFile, myText))
    {
        sscanf(myText.c_str(), "%s %lf", tempC, &tempD);
        if (strcmp(tempC, "nD:") == 0)
        {
            opts->nD = (int)tempD;
        }
        else if (strcmp(tempC, "numDC:") == 0)
        {
            opts->numDC = (int)tempD;
            // allocate the space in memory
            opts->DC = (double *)malloc(opts->numDC * sizeof(double));
            opts->DC_TH = (unsigned char *)malloc(opts->numDC * sizeof(char));
            // set memory
            memset(opts->DC, 0, opts->numDC * sizeof(double));
            memset(opts->DC_TH, 0, opts->numDC * sizeof(char));
            DC_read++;
            DC_TH_read++;
        }
        else if (strcmp(tempC, tempDC) == 0)
        {
            opts->DC[DC_read - 1] = tempD;
            DC_read++;
        }
        else if (strcmp(tempC, tempDC_TH) == 0)
        {
            opts->DC_TH[DC_TH_read - 1] = (unsigned char)tempD;
            DC_TH_read++;
        }
        else if (strcmp(tempC, "MeshAmpX:") == 0)
        {
            opts->MeshIncreaseX = (int)tempD;
        }
        else if (strcmp(tempC, "MeshAmpY:") == 0)
        {
            opts->MeshIncreaseY = (int)tempD;
        }
        else if (strcmp(tempC, "MeshAmpZ:") == 0)
        {
            opts->MeshIncreaseZ = (int)tempD;
        }
        else if (strcmp(tempC, "InputName:") == 0)
        {
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            strcpy(opts->inputFilename, tempFilenames);
        }
        else if (strcmp(tempC, "OutputName:") == 0)
        {
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            strcpy(opts->outputFilename, tempFilenames);
        }
        else if (strcmp(tempC, "printCMap:") == 0)
        {
            opts->printCmap = (int)tempD;
        }
        else if (strcmp(tempC, "CMapName:") == 0)
        {
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            strcpy(opts->CMapName, tempFilenames);
        }
        else if (strcmp(tempC, "Convergence:") == 0)
        {
            opts->ConvergeCriteria = tempD;
        }
        else if (strcmp(tempC, "MaxIter:") == 0)
        {
            opts->MAX_ITER = (long int)tempD;
        }
        else if (strcmp(tempC, "Verbose:") == 0)
        {
            opts->verbose = (int)tempD;
        }
        else if (strcmp(tempC, "RunBatch:") == 0)
        {
            opts->BatchFlag = (int)tempD;
        }
        else if (strcmp(tempC, "NumImages:") == 0)
        {
            opts->NumImgBatch = (int)tempD;
        }
        else if (strcmp(tempC, "CL:") == 0)
        {
            opts->CLeft = tempD;
        }
        else if (strcmp(tempC, "CR:") == 0)
        {
            opts->CRight = tempD;
        }
        else if (strcmp(tempC, "SS:") == 0)
        {
            opts->SteadyStateFlag = (char)tempD;
        }
        else if (strcmp(tempC, "width:") == 0)
        {
            opts->width = (int)tempD;
        }
        else if (strcmp(tempC, "height:") == 0)
        {
            opts->height = (int)tempD;
        }
        else if (strcmp(tempC, "depth:") == 0)
        {
            opts->depth = (int)tempD;
        }
        else if (strcmp(tempC, "inputType:") == 0)
        {
            opts->inputType = (char)tempD;
        }
        else if (strcmp(tempC, "printOutput:") == 0)
        {
            opts->printOut = (int)tempD;
        }
        else if (strcmp(tempC, "nThreads:") == 0)
        {
            opts->nThreads = (int)tempD;
        }
        else if (strcmp(tempC, "useGPU:") == 0)
        {
            opts->useGPU = (int)tempD;
        }
        else if (strcmp(tempC, "nGPU:") == 0)
        {
            opts->nGPU = (int)tempD;
        }
        else if (strcmp(tempC, "tauSim:") == 0)
        {
            opts->tauSim = (int)tempD;
        }
        else if (strcmp(tempC, "POI_LB:") == 0)
        {
            opts->POI_B[0] = (unsigned char)tempD;
        }
        else if (strcmp(tempC, "POI_UB:") == 0)
        {
            opts->POI_B[1] = (unsigned char)tempD;
        }
        else if (strcmp(tempC, "printFMap:") == 0)
        {
            opts->printFmap = (int)tempD;
        }
        else if (strcmp(tempC, "FMapName:") == 0)
        {
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            strcpy(opts->FMapName, tempFilenames);
        }
        else if (strcmp(tempC, "TF:") == 0)
        {
            opts->TF_Flag = (int)tempD;
        }
        else if (strcmp(tempC, "Charge:") == 0)
        {
            opts->charge = (int)tempD;
        }
        else if (strcmp(tempC, "Current:") == 0)
        {
            opts->current = tempD;
        }
        else if (strcmp(tempC, "Time:") == 0)
        {
            opts->Time = tempD;
        }
        else if (strcmp(tempC, "CD_Time:") == 0)
        {
            opts->cd_time = tempD;
        }
        else if (strcmp(tempC, "Relax_Time:") == 0)
        {
            opts->relaxTime = tempD;
        }
        else if (strcmp(tempC, "StartTime:") == 0)
        {
            opts->StartTime = tempD;
        }
        else if (strcmp(tempC, "StartFlag:") == 0)
        {
            opts->StartMapFlag = (int)tempD;
        }
        else if (strcmp(tempC, "InitCmap:") == 0)
        {
            sscanf(myText.c_str(), "%s %s", tempC, tempFilenames);
            strcpy(opts->StartMapName, tempFilenames);
        }

        // Update the number of expected diffusion coefficients and thresholding for image
        // processing

        if (DC_read <= opts->numDC)
            sprintf(tempDC, "D%d:", DC_read);
        if (DC_TH_read <= opts->numDC)
            sprintf(tempDC_TH, "D_TH%d:", DC_TH_read);
    }
    return;
}