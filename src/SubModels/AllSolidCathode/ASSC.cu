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

    mesh.dt = growth_factor * pow(mesh.dx, 2)/ oASSC.POI_DC;

    // calculate AM VF and active surface area

    ASSC_AM_VF(&mesh, &oASSC, simData);

    activeSA_2D_ASSC(&mesh, &oASSC, simData);

    if (opts.verbose)
        printInputASSC(&opts, &oASSC, &mesh);

    test_funct();

    return 0;
}