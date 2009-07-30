
#include <GL/glew.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

extern "C" 
void meanShiftFilter(dim3, dim3, float*, float*, unsigned int, unsigned int,
					 unsigned int, unsigned int);

// EDISON //////////////////////////////////////////////////////////////////
//include local and system libraries and definitions

#include "rgbluv.h"
#include "meanshiftfilter_common.h"

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>

extern float sigmaS;
extern float sigmaR;
extern float minRegion;

// EDISON //////////////////////////////////////////////////////////////////
unsigned int width;
unsigned int height;

unsigned int * h_img = NULL; 
unsigned int * h_filt = NULL;
unsigned int * h_segm = NULL;
unsigned char * h_bndy = NULL;

float * h_src = NULL; // luv source data
float * h_dst = NULL; // luv manipulated data

float * d_src = NULL; // device source data
float * d_dst = NULL; // device manipulated data
float * h_tmp = NULL; // result from device 

const unsigned int FILT = 0;
const unsigned int SEGM = 1;
const unsigned int BNDY = 2;
const unsigned int APPD = 3;

std::string image = "source.ppm";
std::string path = "../../../src/MeanShift/data/";

std::string imgOutGOLD[] = {
    path + "filtimage_gold.ppm",
    path + "segmimage_gold.ppm",
    path + "bndyimage_gold.ppm",
    path + "appd_fsb_gold.ppm"
};

std::string imgOutCUDA[] = {
	path + "filtimage_cuda.ppm",
    path + "segmimage_cuda.ppm",
    path + "bndyimage_cuda.ppm",
    path + "appd_fsb_cuda.ppm"
};

std::string imgRef[] = {
    path + "filtimage_ref.ppm",
    path + "segmimage_ref.ppm",
    path + "bndyimage_ref.pgm",
    path + "appd_fsb_ref.ppm"
};

std::string imgDiff[] = {
    path + "filtimage_diff.ppm",
    path + "segmimage_diff.ppm",
    path + "bndyimage_diff.pgm",
    path + "appd_fsb_diff.ppm"
};
// Used for constants needed by the kernel
// e.g. sigmaS, sigmaR ...
float h_args[MAX_ARGS];




extern void connect();
extern void boundaries();
extern void computeGold(void);



void loadImageData(int argc, char **argv)
{
    // load image (needed so we can get the width and height before we create the window
    char* image_path = cutFindFilePath(image.c_str(), argv[0]);
    if (image_path == 0) {
        fprintf(stderr, "Error finding image file '%s'\n", image.c_str());
        exit(EXIT_FAILURE);
    }
	printf("Image path %s\n", image_path);
	
    cutilCheckError(cutLoadPPM4ub(image_path, (unsigned char **) &h_img, &width, &height));
	
    if (!h_img) {
        printf("Error opening file '%s'\n", image_path);
        exit(-1);
    }
    printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);
}

void computeCUDA();

int main( int argc, char** argv) 
{	
	sigmaS = 7.0f;
	sigmaR = 6.5f;
	minRegion = 20.0f;
	
	loadImageData(argc, argv);

	extern unsigned int N;
	extern unsigned int L;
	
	N = 3;
	L = height * width;

	
	//Allocate memory for h_dst (filtered image output)
	h_dst = new float[3 /*N*/ * height * width];
	h_src = new float[3 /*N*/ * height * width];
	
	h_filt = new unsigned int [height * width * sizeof(GLubyte) * 4];
	h_segm = new unsigned int [height * width * sizeof(GLubyte) * 4];
	h_bndy = new unsigned char [height * width];
	
	// Prepare the RGB data 
	for(unsigned int i = 0; i < L; i++) {
		extern unsigned int * h_img;
		unsigned char * pix = (unsigned char *)&h_img[i];
		RGBtoLUV(pix, &h_src[N * i]);
	}
	
	std::string append = "convert +append ";
	std::string compare = "compare ";
	std::string open = "open " + imgDiff[APPD];
	
	if (cutCheckCmdLineFlag(argc, (const char**)argv, "gold")) {
		computeGold();
		cutilCheckError(cutSavePPM4ub(imgOutGOLD[FILT].c_str(), (unsigned char *)h_filt, width, height));	
		cutilCheckError(cutSavePPM4ub(imgOutGOLD[SEGM].c_str(), (unsigned char *)h_segm, width, height));
		cutilCheckError(cutSavePGMub( imgOutGOLD[BNDY].c_str(), (unsigned char *)h_bndy, width, height));
		append += imgOutGOLD[FILT] + " ";
		append += imgOutGOLD[SEGM] + " ";
		append += imgOutGOLD[BNDY] + " ";
		append += imgOutGOLD[APPD];
		compare += imgOutGOLD[APPD] + " " + imgRef[APPD] + " " + imgDiff[APPD];
		
	} else {
		computeCUDA();
		cutilCheckError(cutSavePPM4ub(imgOutCUDA[FILT].c_str(), (unsigned char *)h_filt, width, height));	
		cutilCheckError(cutSavePPM4ub(imgOutCUDA[SEGM].c_str(), (unsigned char *)h_segm, width, height));
		cutilCheckError(cutSavePGMub( imgOutCUDA[BNDY].c_str(), (unsigned char *)h_bndy, width, height));
		append += imgOutCUDA[FILT] + " ";
		append += imgOutCUDA[SEGM] + " ";
		append += imgOutCUDA[BNDY] + " ";
		append += imgOutCUDA[APPD];

		compare += imgOutCUDA[APPD] + " " + imgRef[APPD] + " " + imgDiff[APPD];
	}
	
	system(append.c_str());
	system(compare.c_str());
	system(open.c_str());

			
	exit(EXIT_SUCCESS);
}

#define GLOBAL_MEMORY 1       // use global memory of the GPU

void computeCUDA() 
{
	unsigned int thx = 32;
	unsigned int thy = 4;
	
	unsigned int imgSize = height * width * sizeof(float) * 3;
	
	cutilSafeCall(cudaMalloc((void**) &d_src, imgSize));
	cutilSafeCall(cudaMalloc((void**) &d_dst, imgSize));
	
	// convert to float array and then copy ... 
	// if we use textures omit this step 
	
#ifdef GLOBAL_MEMORY
	float * h_flt = new float[imgSize];
	// we need here h_src the converted rgb data not h_img the plain rgb!!
	for (unsigned int i = 0; i < 3 * height * width; i++) {
		h_flt[i] = h_src[i];
	}
#endif
	
	// copy host memory to device
    cutilSafeCall(cudaMemcpy(d_src, h_flt, imgSize, cudaMemcpyHostToDevice));
	
	// create and start timer
    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

	// copy 
	
	
	// setup execution parameters
	dim3 threads(thx, thy); // 128 threads 
	dim3 grid(width/thx, height/thy);
	
	meanShiftFilter(grid, threads, d_src, d_dst, width, height, sigmaS, sigmaR);
	
	cutilCheckMsg("Kernel Execution failed");
	
	
	// copy result from device to host
	//h_tmp = new float[imgSize];
	cutilSafeCall(cudaMemcpy(h_dst, d_dst, imgSize, cudaMemcpyDeviceToHost));
	
	for(unsigned int i = 0; i < width * height; i++) {
		unsigned char * pix = (unsigned char *)&h_filt[i];
		LUVtoRGB(&h_dst[3 * i], pix);
	} 
	
	connect();
	boundaries();
		
	// stop and destroy timer
    cutilCheckError(cutStopTimer(timer));
    printf("Processing time: %f (ms) \n", cutGetTimerValue(timer));
    cutilCheckError(cutDeleteTimer(timer));

	// clean up memory
    cutilSafeCall(cudaFree(d_src));
    cutilSafeCall(cudaFree(d_dst));

    cudaThreadExit();
}

