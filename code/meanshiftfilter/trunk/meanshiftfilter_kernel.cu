#ifndef _MSFILTER_KERNEL_H_
#define _MSFILTER_KERNEL_H_

#include <stdio.h>
#include <cutil_inline.h>
#include "meanshiftfilter_common.h"


__constant__ float d_options[MAX_OPTS];

__constant__ float EPSILON = 0.01f;	// define threshold (approx. Value of Mh at a peak or plateau)
__constant__ float LIMIT   = 100.0f;	// define max. # of iterations to find mode
__constant__ unsigned int N = 4;

// declare texture reference for 2D float texture
texture<uchar4, 2, cudaReadModeNormalizedFloat> tex;

__device__ void filter(float4* d_src, float4* d_dst, 
			unsigned int width, unsigned int height,
			float sigmaS, float sigmaR,
			float rsigmaS, float rsigmaR)
			
{
	//Declare variables
	int iterationCount;


	float j, k;	
	float diff0, diff1;
	float dx, dy, dl, du, dv;

	float mvAbs;
	float wsum;
	
	// Traverse each data point applying mean shift
	// to each data point
	float yk[5];
	float Mh[5];
	
	
	float ix = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float iy = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
	
	// Assign window center (window centers are
	// initialized by createLattice to be the point
	// data[i])	
	float4 luv = tex2D(tex, iy, ix); 	// float4 luv = d_src[i];

	yk[0] = iy;
	yk[1] = ix;
	yk[2] = luv.x; // l
	yk[3] = luv.y; // u
	yk[4] = luv.z; // v


	// Calculate its magnitude squared
	mvAbs = 1.0f;
	// Initialize mean shift vector
	Mh[0] = 0.0f;
	Mh[1] = 0.0f;
	Mh[2] = 0.0f;
	Mh[3] = 0.0f;
	Mh[4] = 0.0f;

	
	// Keep shifting window center until the magnitude squared of the
	// mean shift vector calculated at the window center location is
	// under a specified threshold (Epsilon)
	
	// NOTE: iteration count is for speed up purposes only - it
	//       does not have any theoretical importance
	iterationCount = 1;

	while((mvAbs >= EPSILON) && (iterationCount < LIMIT))
	{
		// Shift window location
		yk[0] += Mh[0];
		yk[1] += Mh[1];
		yk[2] += Mh[2];
		yk[3] += Mh[3];
		yk[4] += Mh[4];
		
		// Calculate the mean shift vector at the new
		// window location using lattice

		// Initialize mean shift vector
		Mh[0] = 0.0f;
		Mh[1] = 0.0f;
		Mh[2] = 0.0f;
		Mh[3] = 0.0f;
		Mh[4] = 0.0f;
	
		// Initialize wsum
		wsum = 0.0f;
	
		// Perform lattice search summing
		// all the points that lie within the search
		// window defined using the kernel specified
		// by uniformKernel

	
		//Define bounds of lattice...
		//the lattice is a 2dimensional subspace whose
		//search window bandwidth is specified by sigmaS:
		int LowerBoundX = rintf(yk[0] - sigmaS);
		int LowerBoundY = rintf(yk[1] - sigmaS);
		int UpperBoundX = yk[0] + sigmaS;
		int UpperBoundY = yk[1] + sigmaS;
	
		if (UpperBoundX >= width)  UpperBoundX = width - 1;
		if (UpperBoundY >= height) UpperBoundY = height - 1;
	
		//Perform search using lattice
		//Iterate once through a window of size sigmaS
		for(j = LowerBoundY; j <= UpperBoundY; j += 1)
		for(k = LowerBoundX; k <= UpperBoundX; k += 1)
		{
			diff0 = 0;
			diff1 = 0;
		
			//get index into data array
			luv = tex2D(tex, k, j); //luv = d_src[i * width + j];
		
			//Determine if inside search window
			//Calculate distance squared of sub-space s	

			dx = (k - yk[0]) * rsigmaS;
			dy = (j - yk[1]) * rsigmaS;
			dl = (luv.x - yk[2]) * rsigmaR;               
			du = (luv.y - yk[3]) * rsigmaR;               
			dv = (luv.z - yk[4]) * rsigmaR;               

			diff0 += dx * dx;
			diff0 += dy * dy;

			if((yk[2] > 80.0f)) 
				diff1 += 4.0f * dl * dl;
			else
				diff1 += dl * dl;

			diff1 += du * du;
			diff1 += dv * dv;

		
			// If its inside search window perform sum and count
			// For a uniform kernel weight == 1 for all feature points
			if((diff0 < 1.0f && diff1 < 1.0f))
			{
				// considered point is within sphere => accumulate to mean
				Mh[0] += k;
				Mh[1] += j;
				Mh[2] += luv.x;
				Mh[3] += luv.y;
				Mh[4] += luv.z;
				wsum += 1.0f; //weight
			}
		}

	
		// When using uniformKernel wsum is always > 0 .. since weight == 1 and 
		// wsum += weight. @see uniformLSearch for details ...
	
		// determine the new center and the magnitude of the meanshift vector
		// meanshiftVector = newCenter - center;
		wsum = 1.0f/wsum; 
		
		Mh[0] = Mh[0] * wsum - yk[0];
		Mh[1] = Mh[1] * wsum - yk[1];
		Mh[2] = Mh[2] * wsum - yk[2];
		Mh[3] = Mh[3] * wsum - yk[3];
		Mh[4] = Mh[4] * wsum - yk[4];


		
		// Calculate its magnitude squared
		mvAbs = 0;
		mvAbs += Mh[0] * Mh[0];
		mvAbs += Mh[1] * Mh[1];
		mvAbs += Mh[2] * Mh[2];
		mvAbs += Mh[3] * Mh[3];
		mvAbs += Mh[4] * Mh[4];
		
		// Increment iteration count
		iterationCount += 1;
	}

	
	// Shift window location
	yk[0] += Mh[0];
	yk[1] += Mh[1];
	yk[2] += Mh[2];
	yk[3] += Mh[3];
	yk[4] += Mh[4];
	
	luv = make_float4(yk[2], yk[3], yk[4], 0.0f);

	//__syncthreads();
	// store result into global memory
	int i = ix * width + iy;
	d_dst[i] = luv;
	return;
}


__global__ void mean_shift_filter(float4* d_src, float4* d_dst, 
				unsigned int width, unsigned int height,
				float sigmaS, float sigmaR,
				float rsigmaS, float rsigmaR)
{
	filter(d_src, d_dst, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR);
}


extern "C" void initTexture(int width, int height, void *h_src)
{
	cudaArray* d_array;
	int size = width * height * sizeof(uchar4);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4> ();

	cutilSafeCall(cudaMallocArray(&d_array, &channelDesc, width, height )); 
	cutilSafeCall(cudaMemcpyToArray(d_array, 0, 0, h_src, size, cudaMemcpyHostToDevice));
	
	// set texture parameters
//    tex.addressMode[0] = cudaAddressModeWrap;
//    tex.addressMode[1] = cudaAddressModeWrap;
//    tex.filterMode = cudaFilterModeLinear;
	tex.normalized = 0;	// access without normalized texture coordinates
			// [0, width -1] [0, height - 1]
	
	// bind the array to the texture
 	cutilSafeCall(cudaBindTextureToArray(tex, d_array, channelDesc));
}


extern "C" void setArgs(float* h_options)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_options, h_options, MAX_OPTS * sizeof(float)));
}

extern "C" void meanShiftFilter(dim3 grid, dim3 threads, float4* d_src, float4* d_dst,
					 unsigned int width, unsigned int height,
					float sigmaS, float sigmaR,
					float rsigmaS, float rsigmaR)
{
	mean_shift_filter<<< grid, threads>>>(d_src, d_dst, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR);
}


#endif // #ifndef _MSFILTER_KERNEL_H_
