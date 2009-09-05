
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

extern "C" void initTexture(int, int, void*);
extern "C" void meanShiftFilter(dim3, dim3, float4*, 
		float, float,
		float, float, float, float);
extern "C" void luvToRgb(dim3, dim3, float4*, unsigned int*, unsigned int);
extern "C" void rgbToLuv(dim3, dim3, float4*, unsigned int*, unsigned int);

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
unsigned int * h_iter = NULL; // iterations per thread/pixel
unsigned char * h_bndy = NULL;


int gpu = 1;
int thx = 2;
int thy = 64;


float4 * h_src = NULL; // luv source data
float4 * h_dst = NULL; // luv manipulated data


float4 * d_luv = NULL; // device luv manipulated data
unsigned int * d_rgb = NULL; // device rgb converted data


cudaArray* d_array = NULL;  // texture array for luv data

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


void loadImageData(int argc __attribute__ ((unused)), char **argv)
{
	// load image (needed so we can get the width and height before we create the window
	char* image_path = cutFindFilePath(image.c_str(), argv[0]);
	
	if (image_path == 0) {
		std::cout << "Error finding image file " << image << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Image path " << image_path << std::endl;

	cutilCheckError(cutLoadPPM4ub(image_path, (unsigned char **) &h_img, &width, &height));

	if (!h_img) {
		std::cout << "Error opening file  " << image_path << std::endl;
		exit(-1);
	}
	std::cout<< "Loaded " << image_path << " " 
		<< width << " x " << height << " pixels" << std::endl;
}

void computeCUDA();

void checkCUDAError(const char *msg) { 
	cudaError_t err = cudaGetLastError(); 
	if (cudaSuccess != err) { 
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err)); 
		exit(EXIT_FAILURE); 
	}
}


int main( int argc, char** argv) 
{	
#if CUDART_VERSION < 2020
#error "This CUDART version does not support mapped memory!\n"
#endif
	cudaDeviceProp deviceProp;
	// Get properties and verify device 0 supports mapped memory
	cudaGetDeviceProperties(&deviceProp, 0);
	checkCUDAError("cudaGetDeviceProperties");

	if(!deviceProp.canMapHostMemory) {
		fprintf(stderr, "Device %d cannot map host memory!\n", 0);
	}

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if (cutCheckCmdLineFlag(argc, (const char**)argv, "device")) {
		cutilDeviceInit(argc, argv);
	} else {
		cudaSetDevice(cutGetMaxGflopsDeviceId());
	}
	
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "gpu", &gpu)) {
		std::cout << "Setting gpu: " << gpu << std::endl;
	}

	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "thx", &thx)) {
		std::cout << "Setting thx: " << thx << std::endl;
	}
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "thy", &thy)) {
		std::cout << "Setting thy: " << thy << std::endl;
	}


	sigmaS = 7.0f;
	sigmaR = 6.5f;

	minRegion = 20.0f;

	loadImageData(argc, argv);

	N = 4;
	L = height * width;

	// Set options which are transferred to the device
	h_options[SIGMAS] = sigmaS;
	h_options[SIGMAR] = sigmaR;

	//! Allocate memory for h_dst 
	// The memory returned by this call will be considered as 
	// pinned memory by all CUDA contexts, not just the one 
	// that performed the allocation.
	cudaHostAlloc((void**)&h_dst, sizeof(float4) * height * width, cudaHostAllocPortable);
	checkCUDAError("cudaHostAlloc");
	h_src = new float4[height * width];
	
	
	h_filt = new unsigned int [height * width * sizeof(unsigned char) * 4];
	h_segm = new unsigned int [height * width * sizeof(unsigned char) * 4];
	h_iter = new unsigned int [height * width * sizeof(unsigned char) * 4];
	
	h_bndy = new unsigned char [height * width];
	
	

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
		cutilCheckError(cutSavePPM4ub((path + "itr.ppm").c_str(), (unsigned char *)h_iter, width, height));	

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

	cudaFreeHost(h_dst);
	exit(EXIT_SUCCESS);
}

#define START_TIMER 				\
  unsigned int timer = 0;				\
  cutilCheckError(cutCreateTimer(&timer));		\
  cutilCheckError(cutStartTimer(timer));		\
  
#define STOP_TIMER				\
  cutilCheckError(cutStopTimer(timer));


void computeCUDA() 
{
	unsigned int imgSizeFloat4 = height * width * sizeof(float4);
	unsigned int imgSizeUint = height * width * sizeof(unsigned int);

	// setup execution parameters
	dim3 threads(thx, thy); // 128 threads
	dim3 threads0(16, 16);
	dim3 grid(width/thx, height/thy);

	cutilSafeCall(cudaMalloc((void**) &d_luv, imgSizeFloat4));
	cutilSafeCall(cudaMalloc((void**) &d_rgb, imgSizeUint));
	
	// First we need to convert the RGB data to LUV data 
	// we have an extra kernel here because after much optimizing
	// the less time consuming function become major consumer
	// e.g. LUVtoRGB before 0.1% now 30% ...
	float4 * d_src = NULL; 		// device luv source data
	unsigned int * d_img = NULL;	// device rgb source data
	cutilSafeCall(cudaMalloc((void**) &d_img, imgSizeUint));
	cutilSafeCall(cudaMalloc((void**) &d_src, imgSizeFloat4));
	
	START_TIMER //**********************************************************
	
	cutilSafeCall(cudaMemcpy(d_img, h_img, imgSizeUint, cudaMemcpyHostToDevice));
	
	rgbToLuv(grid, threads0, d_src, d_img, width);
	cutilCheckMsg("rgbToLuv Kernel Execution failed");
	
	
	// TEXTURE Begin: allocate array and copy image data to device
	initTexture(width, height, d_src);
	
	meanShiftFilter(grid, threads, d_luv, width, height,
		        sigmaS, sigmaR, 1.0f/sigmaS, 1.0f/sigmaR);
	cutilCheckMsg("meanShiftFilter Kernel Execution failed");
	
	luvToRgb(grid, threads0, d_luv, d_rgb, width);
	cutilCheckMsg("luvToRgb Kernel Execution failed");
	
	
	// copy result from device to host
	cutilSafeCall(cudaMemcpy(h_dst, d_luv, imgSizeFloat4, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_filt, d_rgb, imgSizeUint, cudaMemcpyDeviceToHost));

	
	STOP_TIMER //**********************************************************

	// Without limit cycle float timeGOLD = 10679.209000;
	float timeGOLD = 10284.756f;
	float timeCUDA = cutGetTimerValue(timer);

	std::cout << "Processing time GOLD: " << timeGOLD << " (ms) " << std::endl;	
	std::cout << "Processing time CUDA: " << timeCUDA << " (ms) " << std::endl;
	std::cout << "Speedup CUDA vs. GOLD: " << timeGOLD/timeCUDA << std::endl;

	cutilCheckError(cutDeleteTimer(timer));
	cutilSafeCall(cudaThreadSynchronize());	

	connect();
	boundaries();


	// clean up memory
	cutilSafeCall(cudaFree(d_src));
	cutilSafeCall(cudaFree(d_luv));
	cutilSafeCall(cudaFreeArray(d_array));

	cudaThreadExit();
}

