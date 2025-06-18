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
        C0[i] = oASSC.C0;   // mol/m^3
    }

    // set BCs
    SetBC_ASSC(&opts, &mesh, &oASSC, simData, BC, BC_Value);

    // discretize

    disc2D_ASSC(&opts, &mesh, &oASSC, DC, Coeff, RHS, C0);

    // solve loop

    mesh.currentTime = oASSC.startTime;

    int step = 0;

    double timeToCheck = oASSC.stepTime;

    test_funct();

    return 0;
}