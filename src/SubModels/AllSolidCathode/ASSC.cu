#include <ASSC.cuh>

int main(int argc, char **argv)
{
    // declare structs
    options opts;
    meshInfo mesh;
    ASSCopts oASSC;

    // read input data

    char inputName[100];
    sprintf(inputName, "inputASSC.txt");

    // Check if file exists

    bool fileExist = false;

    // Check if file exists

    if (FILE *TEST = fopen(inputName, "r"))
    {
        fclose(TEST);
        fileExist = true;
    }

    if (!fileExist)
    {
        printf("Input file not found, returning...\n");
        return 1;
    }

    // read input

    readInputGeneral(inputName, &opts);
    readInputASSC(inputName, &oASSC);

    // Load image to simulate

    char *simData;

    readImg2D(&opts, &mesh, simData);

    // set mesh parameters

    mesh.dx = oASSC.pixelRes / opts.MeshIncreaseX;
    mesh.dy = oASSC.pixelRes / opts.MeshIncreaseY;

    // set dt based on POI_DC

    double growth_factor = 10;

    mesh.dt = growth_factor * pow(mesh.dx, 2) / oASSC.POI_DC;

    // subdomains

    char *subDomain = (char *)malloc(sizeof(char) * mesh.nElements);

    memset(subDomain, 0, sizeof(char) * mesh.nElements);

    ASSC2D_subDomainFF(&mesh, &oASSC, simData, subDomain);

    // char subDomainName[100];

    // sprintf(subDomainName, "before.csv");

    // printSubDomains(&mesh, subDomain, subDomainName);

    if(oASSC.filterP == 1)
    {
        // create arrays based on the number of subdomains
        int *subSize = (int *) malloc(sizeof(int) * oASSC.nSubDomains);
        memset(subSize, 0, sizeof(int) * oASSC.nSubDomains);
        get_SDSize_ASSC2D(&mesh, &oASSC, subSize, subDomain);

        // remove sub-domains that are smaller than 10 voxels
        // (have to modify simData and subDomain)
        filterSD_ASSC2D(&mesh, &oASSC, subSize, subDomain, simData);
        
        // Get subdomains again after simData is modified
        memset(subDomain, 0, sizeof(char) * mesh.nElements);
        ASSC2D_subDomainFF(&mesh, &oASSC, simData, subDomain);

        // sprintf(subDomainName, "after.csv");
        // printSubDomains(&mesh, subDomain, subDomainName);

        // free old subSize
        free(subSize);
    }

    // create arrays based on the number of subdomains
    int *subSize = (int *) malloc(sizeof(int) * oASSC.nSubDomains);

    // calculate AM VF and active surface area

    ASSC_AM_VF(&mesh, &oASSC, simData);

    activeSA_2D_ASSC(&mesh, &oASSC, simData);

    if (opts.verbose)
        printInputASSC(&opts, &oASSC, &mesh);

    // Declare arrays

    int nDiag;

    if(opts.nD == 2)
    {
        nDiag = 5;
    } else if(opts.nD == 3)
    {
        nDiag = 7;
    }

    double *DC = (double *)malloc(sizeof(double) * mesh.nElements);
    double *Conc = (double *)malloc(sizeof(double) * mesh.nElements);
    double *C0 = (double *)malloc(sizeof(double) * mesh.nElements);
    double *RHS = (double *)malloc(sizeof(double) * mesh.nElements);
    double *Coeff = (double *)malloc(sizeof(double) * mesh.nElements * nDiag);
    int *BC = (int *)malloc(sizeof(int) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));
    double *BC_Value = (double *)malloc(sizeof(double) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));

    // Intialize arrays

    memset(DC, 0, sizeof(double) * mesh.nElements);
    memset(Conc, 0, sizeof(double) * mesh.nElements);
    memset(C0, 0, sizeof(double) * mesh.nElements);
    memset(RHS, 0, sizeof(double) * mesh.nElements);
    memset(Coeff, 0, sizeof(double) * mesh.nElements * nDiag);
    memset(BC, 0, sizeof(int) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));
    memset(BC_Value, 0, sizeof(double) * (mesh.numCellsY + 2) * (mesh.numCellsX + 2));
    
    // set DC's

    for (int i = 0; i < mesh.nElements; i++)
    {
        if (simData[i] == oASSC.POI)
        {
            DC[i] = oASSC.POI_DC;
        }
    }

    // initial concentrations

    for (int i = 0; i < mesh.nElements; i++)
    {
        if (DC[i] == 0)
            continue;
        Conc[i] = oASSC.C0; // mol/m^3
    }

    if(oASSC.pristine != 0)
    {
        initC_ASSC2D(&oASSC, &mesh, Conc);
    }

    memcpy(C0, Conc, sizeof(double) * mesh.nElements);

    // if mode == 3, adjust for anomalous diffusion

    if(oASSC.mode == 3)
    {
        if(oASSC2D_AnomDiff(&oASSC, &mesh, DC, C0, simData) == 1)
        {
            return 1;
        }
    }

    // zero sub domains size
    double *subDomainAvgC = (double *)malloc(sizeof(double) * oASSC.nSubDomains);

    memset(subSize, 0, sizeof(int) * oASSC.nSubDomains);
    memset(subDomainAvgC, 0, sizeof(double) * oASSC.nSubDomains);

    // check size and avg C
    subAvgC_ASSC2D(&mesh, &oASSC, C0, subDomain, subSize, subDomainAvgC);

    // set BCs
    SetBC_ASSC(&opts, &mesh, &oASSC, simData, BC, BC_Value);

    // discretize
    disc2D_ASSC(&opts, &mesh, &oASSC, DC, Coeff, RHS, C0, simData, subDomain, subDomainAvgC);

    /*
        GPU Stuff:
    */

    // Declare needed arrays

    double *d_Coeff = NULL;
    double *d_RHS = NULL;
    double *d_Conc = NULL;
    double *d_ConcTemp = NULL;

    // Now we confirm that there is a match in GPUs available and user expectations

    if (opts.useGPU)
    {
        int nDevices;
        cudaGetDeviceCount(&nDevices);

        if (nDevices < 1)
        {
            printf("No CUDA-capable GPU Detected! Exiting...\n");
            return 1;
        }
        else if (nDevices < opts.nGPU)
        {
            printf("User requested %d GPUs, but only %d were detected.\n", opts.nGPU, nDevices);
            printf("Proceeding with %d GPUs\n", nDevices);
            opts.nGPU = nDevices;
        }

        // Initialize the GPU arrays
        initGPU_2DSOR(&d_Coeff, &d_RHS, &d_Conc, &d_ConcTemp, &mesh);
    }

    // solve loop

    mesh.currentTime = oASSC.startTime;

    int step = 0;

    double timeToCheck = oASSC.stepTime;

    // save C(y,t)

    saveCyt_ASSC(&mesh, Conc, step);

    double SOC = 0;
    bool SwitchFlag = 0;

    while (mesh.currentTime <= oASSC.totalTime)
    {
        // Update SOC
        SOC = mesh.currentTime / oASSC.totalTime * 100;

        if(oASSC.C_or_D == 2 && mesh.currentTime >= oASSC.switchTime && SwitchFlag == 0)
        {
            // Change from Charge to Discharge
            oASSC.faceFlux = -oASSC.faceFlux;
            // discretize
            disc2D_ASSC(&opts, &mesh, &oASSC, DC, Coeff, RHS, C0, simData, subDomain, subDomainAvgC);
            // Update Flags
            SwitchFlag = 1;
            mesh.Charging = 0;
            //printf
            if(opts.verbose)
            {
                printf("Switched from Charge to Discharge\n");
                printf("Time = %1.3e\n", mesh.currentTime);
            }
        }

        // not using GITT data
        if (mesh.currentTime != 0)
        {
            // regularize negative concentrations
            fixC_ASSC2D(&mesh, DC, C0);
            if(oASSC.mode == 3)
            {
                // update diffusion coefficients
                if(oASSC2D_AnomDiff(&oASSC, &mesh, DC, C0, simData) == 1)
                    return 1;
                // discretize
                disc2D_ASSC(&opts, &mesh, &oASSC, DC, Coeff, RHS, C0, simData, subDomain, subDomainAvgC);
            }
            else
            {
                // coefficient matrix is still good, just update the RHS
                RHS_Up2D_ASSC(&mesh, &oASSC, DC, Coeff, RHS, C0, simData, subDomain, subDomainAvgC);
            }
        }


        if (opts.useGPU == 0)
        {
            // CPU Solve
            // omp_set_num_threads(opts.nThreads);

            // GS2D_OMP(Coeff, RHS, Conc, &opts, &mesh);
            printf("Not Currently Implemented, returning...\n");
            return 1;
        }
        else
        {
            // GPU Solve
            if (mesh.currentTime == 0)
            {
                JI2D_PBx_GPU(Coeff, RHS, Conc, d_Coeff,
                             d_RHS, d_Conc, d_ConcTemp, &opts, &mesh);
            }
            else
            {
                JI2D_TransientUpdate_PBx(RHS, Conc, d_Coeff,
                                         d_RHS, d_Conc, d_ConcTemp, &opts, &mesh);
            }
        }
        // regularize concentrations

        reg_Conc_ASSC2D(&oASSC, &mesh, Conc);

        // Update time
        mesh.currentTime += mesh.dt;

        // Copy new concentration into C0
        memcpy(C0, Conc, sizeof(double) * mesh.nElements);

        // save data if necessary
        if (mesh.currentTime > timeToCheck)
        {
            timeToCheck += oASSC.stepTime;
            step++;

            // regularize negative concentrations
            fixC_ASSC2D(&mesh, DC, Conc);

            // save avg C(y,t)
            saveCyt_ASSC(&mesh, Conc, step);
            
            if (opts.verbose)
                printf("Time = %1.3e\n", mesh.currentTime);
            
            // check NaN's
            for(int i = 0; i < mesh.nElements; i++)
            {
                if(Conc[i] != Conc[i])
                {
                    printf("Found NaN at %d, time %1.3e\n", i, mesh.currentTime);
                    return 1;
                }
            }
        }
    }

    printCandF_ASSC(&opts, &oASSC, &mesh, DC, Conc);

    // Memory Management
    if(opts.useGPU)
        unInitGPU_SOR(&d_Coeff, &d_RHS, &d_Conc, & d_ConcTemp);

    // simulation arrays
    free(RHS);
    free(DC);
    free(Conc);
    free(C0);
    free(Coeff);

    // morphology data
    free(simData);
    free(subDomain);
    free(subDomainAvgC);
    free(subSize);
    free(BC);
    free(BC_Value);

    return 0;
}