#ifndef _MSFILTER_KERNEL_H_
#define _MSFILTER_KERNEL_H_

#include <stdio.h>
#include <cutil_inline.h>
#include "meanshiftfilter_common.h"

//#define USE_CONST_MEMORY 1

__constant__ float d_options[MAX_OPTS];

__constant__ float EPSILON = 0.01f;	// define threshold (approx. Value of Mh at a peak or plateau)
__constant__ float LIMIT   = 100.0f;	// define max. # of iterations to find mode
__constant__ unsigned int N = 4;

// declare texture reference for 2D float texture
texture<float4, 2, cudaReadModeElementType> tex;

#ifndef USE_CONST_MEMORY
__device__ void uniformSearch(float *Mh, float *yk, float* wsum, float4* d_src, float4* d_dst, 
							  unsigned int width, unsigned int height,
							  float sigmaS, float sigmaR,
							  float rsigmaS, float rsigmaR)
#else
__device__ void uniformSearch(float *Mh, float *yk, float* wsum, float4* d_src, float4* d_dst, 
							  unsigned int width, unsigned int height)

#endif
{
	
	//Declare variables
	int	i, j;
	
	float diff0, diff1;
	float dx, dy, dl, du, dv;
	
	float4 luv; 
	//Define bounds of lattice...
	//the lattice is a 2dimensional subspace whose
	//search window bandwidth is specified by sigmaS:
	
#ifndef USE_CONST_MEMORY
	int LowerBoundX = yk[0] - sigmaS;
	int LowerBoundY = yk[1] - sigmaS;
	int UpperBoundX = yk[0] + sigmaS;
	int UpperBoundY = yk[1] + sigmaS;
#else
	int LowerBoundX = yk[0] - d_options[SIGMAS];
	int LowerBoundY = yk[1] - d_options[SIGMAS];
	int UpperBoundX = yk[0] + d_options[SIGMAS];
	int UpperBoundY = yk[1] + d_options[SIGMAS];
#endif
	
	if (LowerBoundX < 0)
	LowerBoundX = 0;
	if (LowerBoundY < 0)
	LowerBoundY = 0;
	if (UpperBoundX >= width)
	UpperBoundX = width - 1;
	if (UpperBoundY >= height)
	UpperBoundY = height - 1;
	
	//Perform search using lattice
	//Iterate once through a window of size sigmaS
	for(i = LowerBoundY; i <= UpperBoundY; i++)
	for(j = LowerBoundX; j <= UpperBoundX; j++)
	{
		diff0 = 0;
		diff1 = 0;
		
		//get index into data array
		luv = tex2D(tex, j, i); //luv = d_src[i * width + j];
		
		//Determine if inside search window
		//Calculate distance squared of sub-space s	
#ifndef USE_CONST_MEMORY
		dx = (j - yk[0]) / sigmaS;
		dy = (i - yk[1]) / sigmaS;
		dl = (luv.x - yk[2]) / sigmaR;               
		du = (luv.y - yk[3]) / sigmaR;               
		dv = (luv.z - yk[4]) / sigmaR;               
#else
		dx = (j - yk[0]) / d_options[SIGMAS];
		dy = (i - yk[1]) / d_options[SIGMAS];
		dl = (luv.x - yk[2]) / d_options[SIGMAR];               
		du = (luv.y - yk[3]) / d_options[SIGMAR];               
		dv = (luv.z - yk[4]) / d_options[SIGMAR];               
		
#endif
		diff0 += dx * dx;
		diff0 += dy * dy;

		if((yk[2] > 80)) 
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
			Mh[0] += j;
			Mh[1] += i;
			Mh[2] += luv.x;
			Mh[3] += luv.y;
			Mh[4] += luv.z;
			(*wsum) += 1; //weight
		}
	}
	return;
}

#ifndef USE_CONST_MEMORY
__device__ void latticeVector(float *Mh_ptr, float *yk_ptr, float4* d_src, float4* d_dst, 
							  unsigned int width, unsigned int height,
							  float sigmaS, float sigmaR,
							  float rsigmaS, float rsigmaR)
#else
__device__ void latticeVector(float *Mh_ptr, float *yk_ptr, float4* d_src, float4* d_dst, 
							  unsigned int width, unsigned int height)
#endif							  
{

	// Initialize mean shift vector
	
	Mh_ptr[0] = 0.0f;
	Mh_ptr[1] = 0.0f;
	Mh_ptr[2] = 0.0f;
	Mh_ptr[3] = 0.0f;
	Mh_ptr[4] = 0.0f;
	
	// Initialize wsum
	float wsum = 0.0f;
	
	// Perform lattice search summing
	// all the points that lie within the search
	// window defined using the kernel specified
	// by uniformKernel

#ifndef USE_CONST_MEMORY
	uniformSearch(Mh_ptr, yk_ptr, &wsum, d_src, d_dst, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR);
#else
	uniformSearch(Mh_ptr, yk_ptr, &wsum, d_src, d_dst, width, height);
#endif
	
	// When using uniformKernel wsum is always > 0 .. since weight == 1 and 
	// wsum += weight. @see uniformLSearch for details ...
	
	// determine the new center and the magnitude of the meanshift vector
	// meanshiftVector = newCenter - center;
	Mh_ptr[0] = Mh_ptr[0]/wsum - yk_ptr[0];
	Mh_ptr[1] = Mh_ptr[1]/wsum - yk_ptr[1];
	Mh_ptr[2] = Mh_ptr[2]/wsum - yk_ptr[2];
	Mh_ptr[3] = Mh_ptr[3]/wsum - yk_ptr[3];
	Mh_ptr[4] = Mh_ptr[4]/wsum - yk_ptr[4];

}

#ifndef USE_CONST_MEMORY
__device__ void filter(float4* d_src, float4* d_dst, 
					   unsigned int width, unsigned int height,
					   float sigmaS, float sigmaR,
					   float rsigmaS, float rsigmaR)
#else
__device__ void filter(float4* d_src, float4* d_dst, 
					   unsigned int width, unsigned int height)
#endif					   
{
	// Declare Variables
	int   iterationCount;
	float mvAbs;
	
	// Traverse each data point applying mean shift
	// to each data point
	float yk[5];
	float Mh[5];
	
	
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int i = ix * width + iy;
	
	// Assign window center (window centers are
	// initialized by createLattice to be the point
	// data[i])	
	yk[0] = iy;
	yk[1] = ix;
	
	float4 luv = tex2D(tex, iy, ix); 	// float4 luv = d_src[i];
	
	yk[2] = luv.x; // l
	yk[3] = luv.y; // u
	yk[4] = luv.z; // v

	// Calculate the mean shift vector using the lattice
#ifndef USE_CONST_MEMORY
	latticeVector(Mh, yk, d_src, d_dst, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR);
#else
	latticeVector(Mh, yk, d_src, d_dst, width, height);
#endif	

	// Calculate its magnitude squared
	mvAbs = 0;
	
	mvAbs += Mh[0]*Mh[0];
	mvAbs += Mh[1]*Mh[1];
	mvAbs += Mh[2]*Mh[2];
	mvAbs += Mh[3]*Mh[3];
	mvAbs += Mh[4]*Mh[4];
	
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
#ifndef USE_CONST_MEMORY
		latticeVector(Mh, yk, d_src, d_dst, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR);
#else
		latticeVector(Mh, yk, d_src, d_dst, width, height);
#endif	
		
		// Calculate its magnitude squared
		mvAbs = 0;
		mvAbs += Mh[0] * Mh[0];
		mvAbs += Mh[1] * Mh[1];
		mvAbs += Mh[2] * Mh[2];
		mvAbs += Mh[3] * Mh[3];
		mvAbs += Mh[4] * Mh[4];
		
		// Increment iteration count
		iterationCount++;
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
	d_dst[i] = luv;
	return;
}

#ifndef USE_CONST_MEMORY
__global__ void mean_shift_filter(float4* d_src, float4* d_dst, 
								  unsigned int width, unsigned int height,
								  float sigmaS, float sigmaR,
								  float rsigmaS, float rsigmaR)
#else
__global__ void mean_shift_filter(float4* d_src, float4* d_dst, 
								  unsigned int width, unsigned int height)
#endif								  
{
#ifndef USE_CONST_MEMORY
	filter(d_src, d_dst, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR);
#else
	filter(d_src, d_dst, width, height);
#endif
}


extern "C" void initTexture(int width, int height, void *h_flt)
{
    cudaArray* d_array;
    int size = width * height * sizeof(float4);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4> ();

    cutilSafeCall(cudaMallocArray(&d_array, &channelDesc, width, height )); 
    cutilSafeCall(cudaMemcpyToArray(d_array, 0, 0, h_flt, size, cudaMemcpyHostToDevice));
	
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
#ifndef USE_CONST_MEMORY
	mean_shift_filter<<< grid, threads>>>(d_src, d_dst, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR);
#else
	mean_shift_filter<<< grid, threads>>>(d_src, d_dst, width, height);	
#endif
}


#endif // #ifndef _MSFILTER_KERNEL_H_
