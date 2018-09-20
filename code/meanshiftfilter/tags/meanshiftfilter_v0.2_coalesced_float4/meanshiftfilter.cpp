
#include <GL/glew.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_functions.h>
//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
//#include <rendercheck_gl.h>

extern "C" 
void setArgs(float*);
extern "C" 
void meanShiftFilter(dim3, dim3, float4*, float4*, 
					 unsigned int, unsigned int,
					 float, float);

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

extern unsigned int N;
extern unsigned int L;


// EDISON //////////////////////////////////////////////////////////////////
unsigned int width;
unsigned int height;

unsigned int * h_img = NULL; 
unsigned int * h_filt = NULL;
unsigned int * h_segm = NULL;
unsigned char * h_bndy = NULL;

float4 * h_src = NULL; // luv source data
float4 * h_dst = NULL; // luv manipulated data

float4 * d_src = NULL; // device source data
float4 * d_dst = NULL; // device manipulated data

const unsigned int FILT = 0;
const unsigned int SEGM = 1;
const unsigned int BNDY = 2;
const unsigned int APPD = 3;

std::string image = "source.ppm";
std::string path = "/root/data/";

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
float h_options[MAX_OPTS];


extern void connect();
extern void boundaries();
extern void computeGold(void);


void loadImageData(int argc __attribute__ ((unused)),
				   char **argv)
{
    // load image (needed so we can get the width and height before we create the window
    char* image_path = sdkFindFilePath(image.c_str(), argv[0]);
    if (image_path == 0) {
        fprintf(stderr, "Error finding image file '%s'\n", image.c_str());
        exit(EXIT_FAILURE);
    }
	printf("Image path %s\n", image_path);
	
    //checkCudaErrors(sdkLoadPPM4ub(image_path, (unsigned char **) &h_img, &width, &height));
    sdkLoadPPM4ub(image_path, (unsigned char **) &h_img, &width, &height);
	
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
	
	N = 4;
	L = height * width;

	// Set options which are transferred to the device
	h_options[SIGMAS] = sigmaS;
	h_options[SIGMAR] = sigmaR;
	
	//Allocate memory for h_dst (filtered image output)
	h_dst = new float4[height * width];
	h_src = new float4[height * width];
	
	h_filt = new unsigned int [height * width * sizeof(unsigned char) * 4];
	h_segm = new unsigned int [height * width * sizeof(unsigned char) * 4];
	h_bndy = new unsigned char [height * width];
	
	// Prepare the RGB data 
	for(unsigned int i = 0; i < L; i++) {
		extern unsigned int * h_img;
		unsigned char * pix = (unsigned char *)&h_img[i];
		RGBtoLUV(pix, (float*)&h_src[i]);
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
//	if (checkCmdLineFlag(argc, (const char**)argv, "device")) {
//		gpuDeviceInit(argc, argv);
//	} else {
//		cudaSetDevice(gpuGetMaxGflopsDeviceId());
//	}
        cudaSetDevice(0);
	
	std::string append = "convert +append ";
	std::string compare = "compare ";
#ifdef __linux__
	std::string open = "eog " + imgDiff[APPD];
#else	
	std::string open = "open " + imgDiff[APPD];
#endif	
	
	if (checkCmdLineFlag(argc, (const char**)argv, "gold")) {
		computeGold();
		sdkSavePPM4ub(imgOutGOLD[FILT].c_str(), (unsigned char *)h_filt, width, height);	
		sdkSavePPM4ub(imgOutGOLD[SEGM].c_str(), (unsigned char *)h_segm, width, height);
		sdkSavePGM( imgOutGOLD[BNDY].c_str(), (unsigned char *)h_bndy, width, height);
		append += imgOutGOLD[FILT] + " ";
		append += imgOutGOLD[SEGM] + " ";
		append += imgOutGOLD[BNDY] + " ";
		append += imgOutGOLD[APPD];
		compare += imgOutGOLD[APPD] + " " + imgRef[APPD] + " " + imgDiff[APPD];
		
	} else {
		computeCUDA();
		sdkSavePPM4ub(imgOutCUDA[FILT].c_str(), (unsigned char *)h_filt, width, height);	
		sdkSavePPM4ub(imgOutCUDA[SEGM].c_str(), (unsigned char *)h_segm, width, height);
		sdkSavePGM( imgOutCUDA[BNDY].c_str(), (unsigned char *)h_bndy, width, height);
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

//#define _TEXTURE_MEMORY_ 

void computeCUDA() 
{
	unsigned int thx = 32;
	unsigned int thy = 8;
	
	unsigned int imgSize = height * width * sizeof(float4);
	checkCudaErrors(cudaMalloc((void**) &d_src, imgSize));
	checkCudaErrors(cudaMalloc((void**) &d_dst, imgSize));
	
	// convert to float array and then copy ... 
	float4 * h_flt = new float4[height * width];
	// we need here h_src (luv) the converted rgb data not h_img the plain rgb!!
	for (unsigned int i = 0; i < height * width; i++) {
		h_flt[i] = h_src[i];
	}
	
	// copy host memory to device
	checkCudaErrors(cudaMemcpy(d_src, h_flt, imgSize, cudaMemcpyHostToDevice));
	
	setArgs(h_options);
	// create and start timer
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	
	// setup execution parameters
	dim3 threads(thx, thy); // 128 threads 
	dim3 grid(width/thx, height/thy);
	
	meanShiftFilter(grid, threads, d_src, d_dst, width, height, sigmaS, sigmaR);

	
	getLastCudaError("Kernel Execution failed");
		
	// copy result from device to host
	//h_tmp = new float[imgSize];
	checkCudaErrors(cudaMemcpy(h_dst, d_dst, imgSize, cudaMemcpyDeviceToHost));
	
	for(unsigned int i = 0; i < height * width; i++) {
		unsigned char * pix = (unsigned char *)&h_filt[i];
		LUVtoRGB((float*)&h_dst[i], pix);
	} 
	
	connect();
	boundaries();
		
	// stop and destroy timer
	sdkStopTimer(&timer);
	printf("Processing time: %f (ms) \n", sdkGetTimerValue(&timer));
    	sdkDeleteTimer(&timer);

	// clean up memory
	checkCudaErrors(cudaFree(d_src));
	checkCudaErrors(cudaFree(d_dst));

	cudaThreadExit();
}

