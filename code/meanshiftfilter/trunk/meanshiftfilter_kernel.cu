#ifndef _MSFILTER_KERNEL_H_
#define _MSFILTER_KERNEL_H_

#include <stdio.h>
#include <cutil_inline.h>
#include "meanshiftfilter_common.h"


#define EPSILON 0.01f
#define LIMIT 100.0f

// declare texture reference for 2D float texture
texture<float4, 2, cudaReadModeElementType> tex;


__global__ void meanshiftfilter(float4* d_src, float4* d_dst, 
		unsigned int width, unsigned int height,
		float sigmaS, float sigmaR,
		float rsigmaS, float rsigmaR, unsigned int limit)

{
	// NOTE: iteration count is for speed up purposes only - it
	//       does not have any theoretical importance
	int iter = 0;

	float x, y;	
	float diff0, diff1;
	float dx, dy, dl, du, dv;

	volatile float mvAbs;
	float wsum;

	// Traverse each data point applying mean shift
	// to each data point
	float yk[5];
	float Mh[5];

	int ix = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int iy = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

	// Assign window center (window centers are
	// initialized by createLattice to be the point
	// data[i])	
	float4 luv = tex2D(tex, ix, iy); 	// float4 luv = d_src[i];
	
	yk[0] = ix;
	yk[1] = iy;
	yk[2] = luv.x;
	yk[3] = luv.y;
	yk[4] = luv.z;

	// Initialize mean shift vector
	Mh[0] = 0.0f;
	Mh[1] = 0.0f;
	Mh[2] = 0.0f;
	Mh[3] = 0.0f;
	Mh[4] = 0.0f;


	// Keep shifting window center until the magnitude squared of the
	// mean shift vector calculated at the window center location is
	// under a specified threshold (Epsilon)

	volatile float limitcycle[8] = 
	{ 
		12345678.0f,
		12345678.0f, 
		12345678.0f, 
		12345678.0f, 
		12345678.0f, 
		12345678.0f, 
		12345678.0f, 
		12345678.0f 
	}; // Period-4 limit cycle detection

	do {
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

		int lX = yk[0] - sigmaS;
		int lY = yk[1] - sigmaS;
		int uX = yk[0] + sigmaS;
		int uY = yk[1] + sigmaS;


		lX = fmaxf(0.0f, lX);
		lY = fmaxf(0.0f, lY);
		uX = fminf(uX, width - 1);
		uY = fminf(uY, height - 1);

		
		//Perform search using lattice
		//Iterate once through a window of size sigmaS
		for(y = lY; y <= uY; y += 1) {
			for(x = lX; x <= uX; x += 1) {
			
				diff0 = 0.0f;

				//Determine if inside search window
				//Calculate distance squared of sub-space s	

				dx = (x - yk[0]) * rsigmaS;
				dy = (y - yk[1]) * rsigmaS;

				diff0 = dx * dx;
				diff0 += dy * dy;


				if (diff0 >= 1.0f) continue;	
				
				luv = tex2D(tex, x, y); 
				
				diff1 = 0.0f;
				
				dl = (luv.x - yk[2]) * rsigmaR;               
				du = (luv.y - yk[3]) * rsigmaR;               
				dv = (luv.z - yk[4]) * rsigmaR;               


				diff1 = dl * dl;

				if((yk[2] > 80.0f)) 
					diff1 += 3.0f * dl * dl;

				diff1 += du * du;
				diff1 += dv * dv;


				if (diff1 >= 1.0f) continue;


				// If its inside search window perform sum and count
				// For a uniform kernel weight == 1 for all feature points
				// considered point is within sphere => accumulate to mean
				Mh[0] += x;
				Mh[1] += y;
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

		
		
		if (mvAbs == limitcycle[0] || 
		    mvAbs == limitcycle[1] || 
		    mvAbs == limitcycle[2] || 
		    mvAbs == limitcycle[3] ||
		    mvAbs == limitcycle[4] ||
		    mvAbs == limitcycle[5] ||
		    mvAbs == limitcycle[6] ||
		    mvAbs == limitcycle[7]) 
		{
			break;
				
		}

		limitcycle[0] = limitcycle[1];
		limitcycle[1] = limitcycle[2];
		limitcycle[2] = limitcycle[3];
		limitcycle[3] = limitcycle[4];
		limitcycle[4] = limitcycle[5];
		limitcycle[5] = limitcycle[6];
		limitcycle[6] = limitcycle[7];
		limitcycle[7] = mvAbs;
		
		
		
		// Increment iteration count
		iter++;
		
			
	} while((mvAbs >= EPSILON) && (iter < limit));


	// Shift window location
	yk[0] += Mh[0];
	yk[1] += Mh[1];
	yk[2] += Mh[2];
	yk[3] += Mh[3];
	yk[4] += Mh[4];


	luv = make_float4(yk[2], yk[3], yk[4], 0.0f);

	// store result into global memory
	int i = ix + iy * width;
	
	//printf("%d %d \n", i , iter);
	
	//__syncthreads();
	d_dst[i] = luv;

	return;
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


extern "C" void meanShiftFilter(dim3 grid, dim3 threads, float4* d_src, float4* d_dst,
		unsigned int width, unsigned int height,
		float sigmaS, float sigmaR,
		float rsigmaS, float rsigmaR, unsigned int limit)
{
	meanshiftfilter<<< grid, threads>>>(d_src, d_dst, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR, limit);
}


#endif // #ifndef _MSFILTER_KERNEL_H_
