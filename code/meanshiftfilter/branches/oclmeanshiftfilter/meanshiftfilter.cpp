
#include <oclUtils.h>

/*
extern "C" void initTexture(cl_int, cl_int, void*);
extern "C" void meanShiftFilter(dim3, dim3, 
		cl_float4*, cl_float, 
		cl_float, cl_float,
		cl_float, cl_float, 
		cl_float, cl_float);
extern "C" void luvToRgb(dim3, dim3, cl_float4*, cl_uint*, cl_uint);
extern "C" void rgbToLuv(dim3, dim3, cl_float4*, cl_uint*, cl_uint);
*/
// EDISON //////////////////////////////////////////////////////////////////
//include local and system libraries and definitions

#include "rgbluv.h"
#include "meanshiftfilter_common.h"

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <multithreading.h>

#include <iostream>


extern cl_float sigmaS;
extern cl_float sigmaR;
extern cl_float minRegion;


extern cl_uint N;
extern cl_uint L;


// EDISON //////////////////////////////////////////////////////////////////
cl_uint width;
cl_uint height;

cl_uint * h_img = NULL; 
cl_uint * h_filt = NULL;
cl_uint * h_segm = NULL;
cl_uint * h_iter = NULL; // iterations per thread/pixel
cl_uchar * h_bndy = NULL;



cl_int thx = 2;
cl_int thy = 64;
cl_int gpus = 1;


cl_float4 * h_src = NULL; // luv source data
cl_float4 * h_dst = NULL; // luv manipulated data


cl_float4 * d_luv = NULL; // device luv manipulated data
cl_uint * d_rgb = NULL; // device rgb converted data

const cl_uint FILT = 0;
const cl_uint SEGM = 1;
const cl_uint BNDY = 2;
const cl_uint APPD = 3;


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
cl_float h_options[MAX_OPTS];


extern void connect();
extern void boundaries();
extern void computeGold(void);

/*
void loadImageData(cl_int argc, char **argv)
{

	// load image (needed so we can get the width and height before we create the window
	char* image_path = cutFindFilePath(image.c_str(), argv[0]);
	
	if (image_path == 0) {
		std::cout << "Error finding image file " << image << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "Image path " << image_path << std::endl;

	cutilCheckError(cutLoadPPM4ub(image_path, (cl_uchar **) &h_img, &width, &height));

	if (!h_img) {
		std::cout << "Error opening file  " << image_path << std::endl;
		exit(-1);
	}
	std::cout<< "Loaded " << image_path << " " 
		<< width << " x " << height << " pixels" << std::endl;
}

void computeCUDA();
void computeCUDAx();

void checkCUDAError(const char *msg) { 
	cudaError_t err = cudaGetLastError(); 
	if (cudaSuccess != err) { 
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err)); 
		exit(EXIT_FAILURE); 
	}
}
*/


cl_int main( cl_int argc, char** argv) 
{	

	

	/*
#if CUDART_VERSION < 2020
#error "This CUDART version does not support mapped memory!\n"
#endif
	
	cudaDeviceProp deviceProp;
	// Get properties and verify device 0 supports mapped memory
	cudaGetDeviceProperties(&deviceProp, 0);
	checkCUDAError("cudaGetDeviceProperties");

	if(!deviceProp.canMapHostMemory) {
		printf("Device %d cannot map host memory!\n", 0);
	}

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if (cutCheckCmdLineFlag(argc, (const char**)argv, "device")) {
		cutilDeviceInit(argc, argv);
	} else {
		cudaSetDevice(cutGetMaxGflopsDeviceId());
	}
	
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "gpus", &gpus)) {
		std::cout << "Setting gpus: " << gpus << std::endl;
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
	cudaHostAlloc((void**)&h_dst, sizeof(cl_float4) * height * width, cudaHostAllocPortable);
	checkCUDAError("cudaHostAlloc");
	h_src = new cl_float4[height * width];
	
	
	h_filt = new cl_uint [height * width * sizeof(cl_uchar) * 4];
	h_segm = new cl_uint [height * width * sizeof(cl_uchar) * 4];
	h_iter = new cl_uint [height * width * sizeof(cl_uchar) * 4];
	
	h_bndy = new cl_uchar [height * width];
	
	

	std::string append = "convert +append ";
	std::string compare = "compare ";
#ifdef __linux__
	std::string open = "eog " + imgDiff[APPD];
#else	
	std::string open = "open " + imgDiff[APPD];
#endif	

	if (cutCheckCmdLineFlag(argc, (const char**)argv, "gold")) {
		computeGold();
		cutilCheckError(cutSavePPM4ub(imgOutGOLD[FILT].c_str(), (cl_uchar *)h_filt, width, height));	
		cutilCheckError(cutSavePPM4ub(imgOutGOLD[SEGM].c_str(), (cl_uchar *)h_segm, width, height));
		cutilCheckError(cutSavePGMub( imgOutGOLD[BNDY].c_str(), (cl_uchar *)h_bndy, width, height));
		append += imgOutGOLD[FILT] + " ";
		append += imgOutGOLD[SEGM] + " ";
		append += imgOutGOLD[BNDY] + " ";
		append += imgOutGOLD[APPD];
		compare += imgOutGOLD[APPD] + " " + imgRef[APPD] + " " + imgDiff[APPD];

		// iteration count for runtime for each thread
		cutilCheckError(cutSavePPM4ub((path + "itr.ppm").c_str(), (cl_uchar *)h_iter, width, height));	

	} else if (gpus == 1) {
		std::cout << "Using " << gpus << " GPU(s) for segmentation." << std::endl;
		computeCUDA();
		cutilCheckError(cutSavePPM4ub(imgOutCUDA[FILT].c_str(), (cl_uchar *)h_filt, width, height));	
		cutilCheckError(cutSavePPM4ub(imgOutCUDA[SEGM].c_str(), (cl_uchar *)h_segm, width, height));
		cutilCheckError(cutSavePGMub( imgOutCUDA[BNDY].c_str(), (cl_uchar *)h_bndy, width, height));
		append += imgOutCUDA[FILT] + " ";
		append += imgOutCUDA[SEGM] + " ";
		append += imgOutCUDA[BNDY] + " ";
		append += imgOutCUDA[APPD];
		
		compare += imgOutCUDA[APPD] + " " + imgRef[APPD] + " " + imgDiff[APPD];
		
	} else if (gpus == 2) {
		std::cout << "Using " << gpus << " GPU(s) for segmentation." << std::endl;
		computeCUDAx();
		cutilCheckError(cutSavePPM4ub(imgOutCUDA[FILT].c_str(), (cl_uchar *)h_filt, width, height));	
		cutilCheckError(cutSavePPM4ub(imgOutCUDA[SEGM].c_str(), (cl_uchar *)h_segm, width, height));
		cutilCheckError(cutSavePGMub( imgOutCUDA[BNDY].c_str(), (cl_uchar *)h_bndy, width, height));
		append += imgOutCUDA[FILT] + " ";
		append += imgOutCUDA[SEGM] + " ";
		append += imgOutCUDA[BNDY] + " ";
		append += imgOutCUDA[APPD];
		
		compare += imgOutCUDA[APPD] + " " + imgRef[APPD] + " " + imgDiff[APPD];
	} else {
		std::cout << "Too many GPU(s) for segmentation." << std::endl;
	}
	

	if (cutCheckCmdLineFlag(argc, (const char**)argv, "ref")) { 
		system(append.c_str());
		system(compare.c_str());
		system(open.c_str());
	}

	cudaFreeHost(h_dst);
	exit(EXIT_SUCCESS);
	*/
}

/*
#define START_TIMER 					\
  cl_uint timer = 0;				\
  cutilCheckError(cutCreateTimer(&timer));		\
  cutilCheckError(cutStartTimer(timer));		\
  
#define STOP_TIMER					\
  cutilCheckError(cutStopTimer(timer));


//! computeCUDA uses one GPU for doing mean shift segmentation
//! @see computeCUDAx for a GPU parallel version 
void computeCUDA() 
{
	cl_uint imgSizeFloat4 = height * width * sizeof(cl_float4);
	cl_uint imgSizeUint = height * width * sizeof(cl_uint);

	// setup execution parameters
	dim3 threads(thx, thy); // 128 threads
	dim3 threads0(16, 16);
	dim3 grid(width/thx, height/thy/2);
	dim3 grid0(width/16, height/16);

	cutilSafeCall(cudaMalloc((void**) &d_luv, imgSizeFloat4));
	cutilSafeCall(cudaMalloc((void**) &d_rgb, imgSizeUint));
	
	// First we need to convert the RGB data to LUV data 
	// we have an extra kernel here because after much optimizing
	// the less time consuming function become major consumer
	// e.g. LUVtoRGB before 0.1% now 30% ...
	cl_float4 * d_src = NULL; 		// device luv source data
	cl_uint * d_img = NULL;	// device rgb source data
	cutilSafeCall(cudaMalloc((void**) &d_img, imgSizeUint));
	cutilSafeCall(cudaMalloc((void**) &d_src, imgSizeFloat4));
		
	
	cutilSafeCall(cudaMemcpy(d_img, h_img, imgSizeUint, cudaMemcpyHostToDevice));
	rgbToLuv(grid0, threads0, d_src, d_img, width);
	cutilCheckMsg("rgbToLuv Kernel Execution failed");
	
	// TEXTURE Begin: allocate array and copy image data device to device
	initTexture(width, height, d_src);
	
#define ITER 10.0f
	START_TIMER //**********************************************************
	for (cl_int i = 0; i < ITER; i++)
	meanShiftFilter(grid, threads, d_luv, 0, width, height,
		        sigmaS, sigmaR, 1.0f/sigmaS, 1.0f/sigmaR);
	
	cutilCheckMsg("meanShiftFilter Kernel Execution failed");
	cutilSafeCall(cudaThreadSynchronize());
	
	STOP_TIMER //**********************************************************
	
	luvToRgb(grid0, threads0, d_luv, d_rgb, width);
	cutilCheckMsg("luvToRgb Kernel Execution failed");
	
	// copy result from device to host
	cutilSafeCall(cudaMemcpy(h_dst, d_luv, imgSizeFloat4, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_filt, d_rgb, imgSizeUint, cudaMemcpyDeviceToHost));


	// Without limit cycle cl_float timeGOLD = 10679.209000;
	cl_float timeGOLD = 10284.756f;
	cl_float timeCUDA = cutGetTimerValue(timer) / ITER;

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
	cutilSafeCall(cudaFree(d_img));
	cutilSafeCall(cudaFree(d_rgb));
	cudaThreadExit();

}



const cl_uint GPU_COUNT = 2;

  
//! struct for passing the arguments to each GPU thread 
typedef struct {
	cl_int gpus;
	cl_int device;		// device id
	cl_float time;		// device time
	cl_uint * h_img; 	// host source image data
	cl_float4 * h_dst; 	// host destination image data
	cl_uint * h_filt; 	// host filtered image data
	cl_uint fraction; 	// used to calc the right fractio for each gpu
	cl_uint img_size_float4; // each gpu gets a fraction of the image
	cl_uint img_size_uint;   // each gpu gets a fraction of the image
	
} GPUplan;

//! Each gpu is executing the following procedure
static CUT_THREADPROC gpuThread(GPUplan *p) {
	
		
	cutilSafeCall(cudaSetDevice(p->device));
		
	// setup execution parameters
	cl_uint gx = width/thx;
	cl_uint gy = height/thy;

	cl_uint thx0 = 16;	// best configuration for rgb & luv kernel
	cl_uint thy0 = 16;	// best configuration for rgb & luv kernel
	
	cl_float4 * d_luv = NULL;
	cl_uint * d_rgb = NULL;
	
	cutilSafeCall(cudaMalloc((void**) &d_luv, p->img_size_float4));
	cutilSafeCall(cudaMalloc((void**) &d_rgb, p->img_size_uint));
	
	// First we need to convert the RGB data to LUV data 
	// we have an extra kernel here because after much optimizing
	// the less time consuming function become major consumer
	// e.g. LUVtoRGB before 0.1% now 30% ...
	cl_float4 * d_src = NULL; 		// device luv source data
	cl_uint * d_img = NULL;	// device rgb source data
	cutilSafeCall(cudaMalloc((void**) &d_img, p->img_size_uint));
	cutilSafeCall(cudaMalloc((void**) &d_src, p->img_size_float4));

	gx = width/thx0;
	gy = height/thy0;
	dim3 rgbGrid(gx, gy);
	dim3 rgbThrd(thx0, thy0);
	
	// since we do not know where the ms vector is moving each gpus gets 
	// a complete image in luv color space. 
	cutilSafeCall(cudaMemcpy(d_img, h_img, p->img_size_uint, cudaMemcpyHostToDevice));
	rgbToLuv(rgbGrid, rgbThrd, d_src, d_img, width);
	cutilCheckMsg("rgbToLuv Kernel Execution failed");
	
	//  allocate array and copy image data device to device
	initTexture(width, height, d_src);
	
	// now we have the complete image in luv color space 
	// the ms filter can now move on the complete image
	
	// but for a multiple gpu configuration each gpu gets its
	// own segment to operate on , therefore move the d_luv pointer.
	cl_float4 		* d_luv0 = d_luv + p->fraction * p->device; 
	cl_uint 	* d_rgb0 = d_rgb + p->fraction * p->device;
	
	// the ms filter will only filter half of an image
	gx = width/thx;
	gy = height/thy/p->gpus;
	dim3 msGrid(gx, gy);
	dim3 msThrd(thx, thy);

	// offset needed for texture access cl_float4 texture element
	cl_float offset = height/p->gpus * p->device;
		
#define ITER 1.0f
	START_TIMER //**********************************************************

	for (cl_int i = 0; i < ITER; i++)
	meanShiftFilter(msGrid, msThrd, d_luv0, offset, width, height,
		        sigmaS, sigmaR, 1.0f/sigmaS, 1.0f/sigmaR);
	
	cutilCheckMsg("meanShiftFilter Kernel Execution failed");
	cutilSafeCall(cudaThreadSynchronize());
	
	STOP_TIMER //**********************************************************

		
	// the luv kernel needs only the half fo the image for 2 gpus
	gx = width/thx0;
	gy = height/thy0/p->gpus;
	dim3 luvGrid(gx, gy);
	dim3 luvThrd(thx0, thy0);
	

	luvToRgb(luvGrid, luvThrd, d_luv0, d_rgb0, width);
	cutilCheckMsg("luvToRgb Kernel Execution failed");
		
	// copy result from device to host
	cutilSafeCall(cudaMemcpy(p->h_dst, d_luv0, p->img_size_float4/p->gpus, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(p->h_filt, d_rgb0, p->img_size_uint/p->gpus, cudaMemcpyDeviceToHost));


	p->time = cutGetTimerValue(timer) / ITER;

	cutilCheckError(cutDeleteTimer(timer));
	cutilSafeCall(cudaThreadSynchronize());	

	// clean up memory
	cutilSafeCall(cudaFree(d_img));
	cutilSafeCall(cudaFree(d_src));
	cutilSafeCall(cudaFree(d_luv));
	cutilSafeCall(cudaFree(d_rgb));
		
	CUT_THREADEND;
}

//! parallel GPU version of mean shift filter. Here we're using two 
//! Geforce 8800 GTS GPUs
void computeCUDAx() 
{
	
	GPUplan plan[GPU_COUNT];
	CUTThread threadID[GPU_COUNT];

	cutilSafeCall(cudaGetDeviceCount(&gpus));
	std::cout << "CUDA capable device count: " << gpus << std::endl;

	
	for (cl_uint i = 0; i < gpus; i++) {
		plan[i].gpus = gpus;
		plan[i].device = i;
		
		plan[i].fraction = height * width / gpus; 
		
		plan[i].h_dst = h_dst + plan[i].fraction * i;
		plan[i].h_filt = h_filt + plan[i].fraction * i;
		
		plan[i].img_size_float4 = height * width * sizeof(cl_float4);
		plan[i].img_size_uint = height * width * sizeof(cl_uint);
	}
	
	for (cl_uint i = 0; i < gpus; i++) {
		threadID[i] = cutStartThread((CUT_THREADROUTINE)gpuThread, (void *)(plan + i));
	}
	cutWaitForThreads(threadID, gpus);
	
	// Without limit cycle cl_float timeGOLD = 10679.209000;
	cl_float timeGOLD = 10284.756f;
	cl_float timeCUDA = 0.0f;
	for (cl_uint i = 0; i < gpus; i++) {
		timeCUDA += plan[i].time;
		std::cout << "GPU" << plan[i].device << " self time: " << plan[i].time << std::endl;
	}
	timeCUDA /= gpus;
	
	std::cout << "CPU" << 0 << ": Processing time GOLD: " << timeGOLD << " (ms) " << std::endl;	
	std::cout << "GPU0+1" << ": Processing time CUDA: " << timeCUDA << " (ms) " << std::endl;
	std::cout << "GPU0+1" << ": Speedup CUDA vs. GOLD: " << timeGOLD/timeCUDA << std::endl;

		
	connect();
	boundaries();

	cudaThreadExit();
}
*/