#include "Deff2D.cuh"

int main(void){

	// Important call for efficiency on Linux
	fflush(stdout);

	//	Declare data structure
	options opts;

	char inputFilename[30];

	sprintf(inputFilename, "input.txt");

	readInputFile(inputFilename, &opts);

	if(opts.BatchFlag == 0){
		SingleSim(opts);
	}

	return 0;

}