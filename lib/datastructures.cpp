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
    char *GITT_Name;        // GITT file name
    int useLinear;          // Linear model for diffusion update as function of concentration
    int useAnom;            // use theory of anomalous diffusion
    double Dprime;          // anomalous diffusion parameter
    double C0;              // initial concentration
    int useMig;             // migration or not (0 = false, 1 = true)
} TSSDopts;

typedef struct
{
    double T;        // operating temperature
    double dE_dL[3]; // array containing the three potentials (dx, dy, and dz, respectively)
} Migration;

typedef struct
{
    int printMAP;          // Will decide if CMaps/FMaps are printed or not
    int POI;               // phase of interest
    int POI_TH;            // grayscale th for POI
    double CMax;           // parameter for concentration
    double D0;             // parameters for anomalous diffusion
    double stepTime;       // time for step (s)
    double totalTime;      // total sim time (s)
    double startTime;      // start time other than 0?
    double currentDensity; // current density (A/m)
    double POI_DC;         // diffusion coefficient of phase of interest
    double C0;             // initial concentration for POI
    double pixelRes;       // pixel resolution
    int PB;                // Periodic BC (0 = false, 1 = true)
    int C_or_D;            // charge (0) or discharge (1)
    int mode;              // controls the type of weighing factor for reaction rate
    double TauE;           // Tortuosity for electorde (needed for mode 1)
    double TauLi;          // SE tortuosity (needed for mode 1)
    int filterP;           // filter small particles or no? (0 = false, 1 = true)
    int filterSizeTH;      // filter subDomains smaller than this
    // not inputs
    double AM_VF;           // volume fraction of active material
    double faceFlux;        // fixed face flux (is the applied current divided by nFaces)
    int nSubDomains;        // number of AM sub-domains
    double TauMax;          // maximum tortuosity
} ASSCopts;

#endif