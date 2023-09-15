#include "Deff2D.cuh"

int main(void){

	fflush(stdout);

	//
	options opts;
	// user input number of threads and default

	char inputFilename[30];

	sprintf(inputFilename, "input.txt");

	readInputFile(inputFilename, &opts);

	return 0;

}