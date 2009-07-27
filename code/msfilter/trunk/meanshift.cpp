
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

// includes, kernels
// #include "meanshiftfilter_kernel.cu"
extern "C" 
void meanShiftFilter(dim3, dim3, float*, float*, unsigned int, unsigned int,
					 unsigned int, unsigned int);

// EDISON //////////////////////////////////////////////////////////////////
//include local and system libraries and definitions
#include "edison.h"
#include "rgbluv.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>

extern float sigmaS;
extern float sigmaR;
extern float minRegion;

// EDISON //////////////////////////////////////////////////////////////////

char *image_filename = "image.ppm";

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

#define SEGM 0
#define FILT 1
#define BNDY 2

const char *imgOut[] = {
"segmimage.ppm",
"filtimage.ppm",
"bndyimage.pgm",
NULL
};

const char *imgRef[] = {
"segmimage0.ppm",
"filtimage0.ppm",
"bndyimage0.pgm",
NULL
};

extern void connect();
extern void boundaries();
extern void computeGold(void);



void loadImageData(int argc, char **argv)
{
    // load image (needed so we can get the width and height before we create the window
    char* image_path = cutFindFilePath(image_filename, argv[0]);
    if (image_path == 0) {
        fprintf(stderr, "Error finding image file '%s'\n", image_filename);
        exit(EXIT_FAILURE);
    }
	
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
	
	if (cutCheckCmdLineFlag(argc, (const char**)argv, "gold")) {
		computeGold();
	} else {
		computeCUDA();
	}
		
	cutilCheckError(cutSavePPM4ub(imgOut[FILT], (unsigned char *)h_filt, width, height));	
	cutilCheckError(cutSavePPM4ub(imgOut[SEGM], (unsigned char *)h_segm, width, height));
	cutilCheckError(cutSavePGMub( imgOut[BNDY], (unsigned char *)h_bndy, width, height));
	
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

