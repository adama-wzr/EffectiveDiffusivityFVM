#ifndef _DATASTRUCT
#define _DATASTRUCT

#include <stdbool.h>
#include <stdlib.h>

typedef struct
{
    int printMAP;           // Will decide if CMaps/FMaps are printed or not
    int POI;                // phase of interest
    double DC_Max;          // Max diffusion coefficient
    double DC_Min;          // min diffusion coefficient
    double DC_Step;         // diffusion coefficient increase step
    double CMax;            // maximum concentration
    double D0;              // standard diffusion coefficient (trace assumption)
    int C_or_D;             // charge (0) or discharge (1)
    double current_density; // current density in A
    double stepSize;        // step time in seconds
    double totalTime;       // total experiment time
    double startTime;       // start time (if not 0)
    double pixelRes;        // pixel resolution in m
    int useGITT;            // flag to use or not GITT (0 = false, 1 = true)
    char* GITT_Name;        // GITT file name
    int useLinear;          // Linear model for diffusion update as function of concentration
    int useAnom;            // use theory of anomalous diffusion
    double Dprime;          // anomalous diffusion parameter
    double C0;              // initial concentration
    int useMig;             // migration or not (0 = false, 1 = true)
} TSSDopts;

typedef struct
{
    double T;               // operating temperature
    double dE_dL[3];        // array containing the three potentials (dx, dy, and dz, respectively) 
} Migration;

#endif