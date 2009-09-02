#ifndef _MSFILTER_KERNEL_H_
#define _MSFILTER_KERNEL_H_

#include <stdio.h>
#include <cutil_inline.h>
#include "meanshiftfilter_common.h"

// declare texture reference for 2D float texture
texture<float4, 2, cudaReadModeElementType> tex;

__global__ void meanshiftfilter(
	float4* d_src, float4* d_dst, 
	float width, float height,
	float sigmaS, float sigmaR,
	float rsigmaS, float rsigmaR)
{

	// NOTE: iteration count is for speed up purposes only - it
	//       does not have any theoretical importance
	float iter = 0;
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

	float limitcycle[8] = 
	{ 
		12345678.0f,
		12345678.0f, 
		12345678.0f, 
		12345678.0f, 
		12345678.0f, 
		12345678.0f, 
		12345678.0f, 
		12345678.0f 
	}; // Period-8 limit cycle detection

	
	float mag;  // magnitude squared
	
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

		float lX = (int)yk[0] - sigmaS;
		float lY = (int)yk[1] - sigmaS;
		float uX = yk[0] + sigmaS;
		float uY = yk[1] + sigmaS;

		lX = fmaxf(0.0f, lX);
		lY = fmaxf(0.0f, lY);
		uX = fminf(uX, width - 1.0f);
		uY = fminf(uY, height - 1.0f);

				
		float x, y;	
		
		//Perform search using lattice
		//Iterate once through a window of size sigmaS
		for(y = lY; y <= uY; y += 1) {
			for(x = lX; x <= uX; x += 1) {

				//Determine if inside search window
				//Calculate distance squared of sub-space s	
				float diff0 = 0.0f;

				//dx = (x - yk[0]) * rsigmaS;
				//dy = (y - yk[1]) * rsigmaS;
				
				float dx_0 = x - yk[0];
				float dy_0 = y - yk[1];
				
				float dx = dx_0 * rsigmaS;
				float dy = dy_0 * rsigmaS;

				float diff0_0 = dx * dx;
				float diff0_1 = dy * dy;
				
				diff0 = diff0_0 + diff0_1;

				if (diff0 >= 1.0f) continue;
				
				luv = tex2D(tex, x, y); 
				
				float diff1 = 0.0f;
				
				//dl = (luv.x - yk[2]) * rsigmaR;               
				//du = (luv.y - yk[3]) * rsigmaR;               
				//dv = (luv.z - yk[4]) * rsigmaR;               
				float dl_0 = luv.x - yk[2];               
				float du_0 = luv.y - yk[3];               
				float dv_0 = luv.z - yk[4];
				
				float dl = dl_0 * rsigmaR; 
				float du = du_0 * rsigmaR;
				float dv = dv_0 * rsigmaR;
				
					
				float diff1_0 = dl * dl;
				float diff1_1 = du * du;
				float diff1_2 = dv * dv;
				diff1 = diff1_0 + diff1_1 + diff1_2;
			
				
				if((yk[2] > 80.0f)) { 
					diff1 += 3.0f * dl * dl;
				}
			
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
		// When using uniform kernel wsum is always > 0 .. since weight == 1 and 
		// wsum += weight. 
		// determine the new center and the magnitude of the meanshift vector
		// meanshiftVector = newCenter - center;
		wsum = 1.0f/wsum; 
		
		//Mh[0] = Mh[0] * wsum - yk[0];
		//Mh[1] = Mh[1] * wsum - yk[1];
		//Mh[2] = Mh[2] * wsum - yk[2];
		//Mh[3] = Mh[3] * wsum - yk[3];
		//Mh[4] = Mh[4] * wsum - yk[4];
		
		float ms_0 = Mh[0] * wsum;
		float ms_1 = Mh[1] * wsum;
		float ms_2 = Mh[2] * wsum;
		float ms_3 = Mh[3] * wsum;
		float ms_4 = Mh[4] * wsum;
		
		Mh[0] = ms_0 - yk[0];
		Mh[1] = ms_1 - yk[1];
		Mh[2] = ms_2 - yk[2];
		Mh[3] = ms_3 - yk[3];
		Mh[4] = ms_4 - yk[4];
		



		// Calculate its magnitude squared
		mag = 0;
		
		float mag_0 = Mh[0] * Mh[0];
		float mag_1 = Mh[1] * Mh[1];
		float mag_2 = Mh[2] * Mh[2];
		float mag_3 = Mh[3] * Mh[3];
		float mag_4 = Mh[4] * Mh[4];
		mag = mag_0 + mag_1 + mag_2 + mag_3 + mag_4; 

		
		// Usually you don't do float == float but in this case
		// it is completely safe as we have limit cycles where the 
		// values after some iterations are equal, the same
		if (mag == limitcycle[0] || 
		    mag == limitcycle[1] || 
		    mag == limitcycle[2] || 
		    mag == limitcycle[3] ||
		    mag == limitcycle[4] ||
		    mag == limitcycle[5] ||
		    mag == limitcycle[6] ||
		    mag == limitcycle[7]) 
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
		limitcycle[7] = mag;
		
				
		// Increment iteration count
		iter += 1;
			
	} while((mag >= EPSILON) && (iter < LIMIT));


	// Shift window location
	yk[0] += Mh[0];
	yk[1] += Mh[1];
	yk[2] += Mh[2];
	yk[3] += Mh[3];
	yk[4] += Mh[4];


	luv = make_float4(yk[2], yk[3], yk[4], 0.0f);

	// store result into global memory
	float i = ix + iy * width;
	d_dst[(int)i] = luv;

	
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
		float width, float height,
		float sigmaS, float sigmaR,
		float rsigmaS, float rsigmaR)
{
	meanshiftfilter<<< grid, threads>>>(d_src, d_dst, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR);
}


#endif // #ifndef _MSFILTER_KERNEL_H_
