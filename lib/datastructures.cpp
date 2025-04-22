#ifndef _DATASTRUCT
#define _DATASTRUCT

#include <stdbool.h>
#include <stdlib.h>

typedef struct
{
    int printCMAP;      // Will decide if CMAPS are printed or not
    int POI;            // phase of interest
    double DC_Max;      // Max diffusion coefficient
    double DC_Min;      // min diffusion coefficient
    double DC_Step;     // diffusion coefficient increase step
    double CMax;        // maximum concentration
    double D0;          // standard diffusion coefficient (trace assumption)
    int C_or_D;         // charge (0) or discharge (1)
    double current_density;     // current density in A
    double stepSize;            // step time in seconds
    double totalTime;           // total experiment time
    double startTime;           // start time (if not 0)
}TSSDopts;


#endif