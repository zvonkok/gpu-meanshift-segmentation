
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

extern "C" void setArgs(float*);
extern "C" void initTexture(int, int, void*);
extern "C" void meanShiftFilter(dim3, dim3, float4*, float4*, 
								unsigned int, unsigned int,
								float, float, float, float);


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
unsigned char * h_iter = NULL; // iterations per thread/pixel


float4 * h_src = NULL; // luv source data
float4 * h_dst = NULL; // luv manipulated data

float4 * d_src = NULL; // device source data
float4 * d_dst = NULL; // device manipulated data

const unsigned int FILT = 0;
const unsigned int SEGM = 1;
const unsigned int BNDY = 2;
const unsigned int APPD = 3;

std::string image = "source.ppm";
std::string path = "../../../src/Meanshift/data/";

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
	h_iter = new unsigned char [height * width];
	
	// Prepare the RGB data 
	for(unsigned int i = 0; i < L; i++) {
		extern unsigned int * h_img;
		unsigned char * pix = (unsigned char *)&h_img[i];
		RGBtoLUV(pix, (float*)&h_src[i]);
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if (cutCheckCmdLineFlag(argc, (const char**)argv, "device")) {
		cutilDeviceInit(argc, argv);
	} else {
		cudaSetDevice(cutGetMaxGflopsDeviceId());
	}
	
	std::string append = "convert +append ";
	std::string compare = "compare ";
#ifdef __linux__
	std::string open = "eog " + imgDiff[APPD];
#else	
	std::string open = "open " + imgDiff[APPD];
#endif	
	
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
		
		// iteration count for runtime for each thread
		cutilCheckError(cutSavePGMub((path + "itr.pgm").c_str(), (unsigned char *)h_iter, width, height));	
		
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
	
	if (cutCheckCmdLineFlag(argc, (const char**)argv, "ref")) { 
		system(append.c_str());
		system(compare.c_str());
		system(open.c_str());
	}

			
	exit(EXIT_SUCCESS);
}

//#define _TEXTURE_MEMORY_ 

void computeCUDA() 
{
	unsigned int thx = 32;
	unsigned int thy = 8;
	
	unsigned int imgSize = height * width * sizeof(float4);
	cutilSafeCall(cudaMalloc((void**) &d_src, imgSize));
	cutilSafeCall(cudaMalloc((void**) &d_dst, imgSize));
	
	// convert to float array and then copy ... 
	float4 * h_flt = new float4[height * width];
	// we need here h_src (luv) the converted rgb data not h_img the plain rgb!!
	for (unsigned int i = 0; i < height * width; i++) {
		h_flt[i] = h_src[i];
	}

	
	// TEXTURE Begin: allocate array and copy image data
	initTexture(width, height, h_flt);
	// TEXTURE End

	
	
	// copy host memory to device
	cutilSafeCall(cudaMemcpy(d_src, h_flt, imgSize, cudaMemcpyHostToDevice));
	
	setArgs(h_options);
	// create and start timer
	unsigned int timer = 0;
	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutStartTimer(timer));
	
	// setup execution parameters
	dim3 threads(thx, thy); // 128 threads 
	dim3 grid(width/thx, height/thy);
	
	std::cout << "params: " << "sigmaS: " << sigmaS << " 1/sigmaS: " << 1.0f/sigmaS << std::endl;
	std::cout << "params: " << "sigmaR: " << sigmaR << " 1/sigmaR: " << 1.0f/sigmaR << std::endl;
	
	meanShiftFilter(grid, threads, d_src, d_dst, width, height, sigmaS, sigmaR, 1.0f/sigmaS, 1.0f/sigmaR);

	
	cutilCheckMsg("Kernel Execution failed");
		
	// copy result from device to host
	//h_tmp = new float[imgSize];
	cutilSafeCall(cudaMemcpy(h_dst, d_dst, imgSize, cudaMemcpyDeviceToHost));
	
	for(unsigned int i = 0; i < height * width; i++) {
		unsigned char * pix = (unsigned char *)&h_filt[i];
		LUVtoRGB((float*)&h_dst[i], pix);
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

