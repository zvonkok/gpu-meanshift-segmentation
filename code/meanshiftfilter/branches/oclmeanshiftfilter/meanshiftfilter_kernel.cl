#ifndef _MSFILTER_KERNEL_H_
#define _MSFILTER_KERNEL_H_

#include <stdio.h>
#include <cutil_inline.h>
#include "meanshiftfilter_common.h"

// declare texture reference for 2D float texture
texture<cl_float4, 2, cudaReadModeElementType> tex;

__global__ void meanshiftfilter(
	cl_float4* d_luv, float offset,
	float width, float height,
	float sigmaS, float sigmaR,
	float rsigmaS, float rsigmaR)
{

	// NOTE: iteration count is for speed up purposes only - it
	//       does not have any theoretical importance
	float iter = 0;
	float wsum;
	

	float ix = blockIdx[0] * blockDim[0] + threadIdx[0];
	float iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	cl_float4 luv = tex2D(tex, ix, iy + offset); 
	
	// Initialize spatial/range vector (coordinates + color values)
	float yj_0 = ix;
	float yj_1 = iy + offset;
	float yj_2 = luv[0];
	float yj_3 = luv[1];
	float yj_4 = luv[2];

	// Initialize mean shift vector
	float ms_0 = 0.0f;
	float ms_1 = 0.0f;
	float ms_2 = 0.0f;
	float ms_3 = 0.0f;
	float ms_4 = 0.0f;

	// Period-8 limit cycle detection
	float lc_0 = 12345678.0f;
	float lc_1 = 12345678.0f;
	float lc_2 = 12345678.0f;
	float lc_3 = 12345678.0f;
	float lc_4 = 12345678.0f;
	float lc_5 = 12345678.0f;
	float lc_6 = 12345678.0f;
	float lc_7 = 12345678.0f;
	 

	// Keep shifting window center until the magnitude squared of the
	// mean shift vector calculated at the window center location is
	// under a specified threshold (Epsilon)
	float mag;  // magnitude squared
	
	do {
		// Shift window location
		yj_0 += ms_0;
		yj_1 += ms_1;
		yj_2 += ms_2;
		yj_3 += ms_3;
		yj_4 += ms_4;


		// Calculate the mean shift vector at the new
		// window location using lattice

		// Initialize mean shift vector
		ms_0 = 0.0f;
		ms_1 = 0.0f;
		ms_2 = 0.0f;
		ms_3 = 0.0f;
		ms_4 = 0.0f;

		// Initialize wsum
		wsum = 0.0f;

		// Perform lattice search summing
		// all the points that lie within the search
		// window defined using the kernel specified
		// by uniformKernel


		//Define bounds of lattice...
		//the lattice is a 2dimensional subspace whose
		//search window bandwidth is specified by sigmaS:

		
		float lX = (int)yj_0 - sigmaS;
		float lY = (int)yj_1 - sigmaS;
		float uX = yj_0 + sigmaS;
		float uY = yj_1 + sigmaS;

		lX = fmaxf(0.0f, lX);
		lY = fmaxf(0.0f, lY);
		uX = fminf(uX, width - 1.0f);
		uY = fminf(uY, height - 1.0f);

				
		
		//Perform search using lattice
		//Iterate once through a window of size sigmaS
		for(float y = lY; y <= uY; y += 1.0f) {
			for(float x = lX; x <= uX; x += 1.0f) {

				//Determine if inside search window
				//Calculate distance squared of sub-space s	
				float diff0 = 0.0f;

				//dx = (x - yj_0) * rsigmaS;
				//dy = (y - yj_1) * rsigmaS;
				
				float dx_0 = x - yj_0;
				float dy_0 = y - yj_1;
				
				float dx = dx_0 * rsigmaS;
				float dy = dy_0 * rsigmaS;

				float diff0_0 = dx * dx;
				float diff0_1 = dy * dy;
				
				diff0 = diff0_0 + diff0_1;

				if (diff0 >= 1.0f) continue;
				
				luv = tex2D(tex, x, y); 
				 
				float diff1 = 0.0f;
	
				float dl_0 = luv[0] - yj_2;               
				float du_0 = luv[1] - yj_3;               
				float dv_0 = luv[2] - yj_4;
				
				float dl = dl_0 * rsigmaR; 
				float du = du_0 * rsigmaR;
				float dv = dv_0 * rsigmaR;
				
					
				float diff1_0 = dl * dl;
				float diff1_1 = du * du;
				float diff1_2 = dv * dv;
				diff1 = diff1_0 + diff1_1 + diff1_2;
			
				if((yj_2 > 80.0f)) { 
					diff1 += 3.0f * dl * dl;
				}
			
				if (diff1 >= 1.0f) continue;

				// If its inside search window perform sum and count
				// For a uniform kernel weight == 1 for all feature points
				// considered point is within sphere => accumulate to mean
				ms_0 += x;
				ms_1 += y;
				ms_2 += luv[0];
				ms_3 += luv[1];
				ms_4 += luv[2];
				wsum += 1.0f; //weight

			}

		}
		// When using uniform kernel wsum is always > 0 .. since weight == 1 and 
		// wsum += weight. 
		// determine the new center and the magnitude of the meanshift vector
		// meanshiftVector = newCenter - center;
		wsum = 1.0f/wsum; 
		
		float ws_0 = ms_0 * wsum;
		float ws_1 = ms_1 * wsum;
		float ws_2 = ms_2 * wsum;
		float ws_3 = ms_3 * wsum;
		float ws_4 = ms_4 * wsum;
		
		ms_0 = ws_0 - yj_0;
		ms_1 = ws_1 - yj_1;
		ms_2 = ws_2 - yj_2;
		ms_3 = ws_3 - yj_3;
		ms_4 = ws_4 - yj_4;
		

		// Calculate its magnitude squared
		float mag_0 = ms_0 * ms_0;
		float mag_1 = ms_1 * ms_1;
		float mag_2 = ms_2 * ms_2;
		float mag_3 = ms_3 * ms_3;
		float mag_4 = ms_4 * ms_4;
		mag = mag_0 + mag_1 + mag_2 + mag_3 + mag_4; 

		
		// Usually you don't do float == float but in this case
		// it is completely safe as we have limit cycles where the 
		// values after some iterations are equal, the same
		if (mag == lc_0 || mag == lc_1 ||
		    mag == lc_2 || mag == lc_3 ||
		    mag == lc_4 || mag == lc_5 || 
		    mag == lc_6 || mag == lc_7) 
		{
			break;			
		}
				
		lc_0 = lc_1;
		lc_1 = lc_2;
		lc_2 = lc_3;
		lc_3 = lc_4;
		lc_4 = lc_5;
		lc_5 = lc_6;
		lc_6 = lc_7;
		lc_7 = mag;
		
				
		// Increment iteration count
		iter += 1;
			
	} while((mag >= EPSILON) && (iter < LIMIT));


	// Shift window location
	yj_0 += ms_0;
	yj_1 += ms_1;
	yj_2 += ms_2;
	yj_3 += ms_3;
	yj_4 += ms_4;


	luv = make_float4(yj_2, yj_3, yj_4, 0.0f);

	// store result into global memory
	int i = ix + iy * width;
	d_luv[i] = luv;

	
	return;
}



extern "C" void initTexture(int width, int height, void *d_src)
{
	cudaArray* d_array;
	int size = width * height * sizeof(cl_float4);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<cl_float4> ();

	cutilSafeCall(cudaMallocArray(&d_array, &channelDesc, width, height )); 
	cutilSafeCall(cudaMemcpyToArray(d_array, 0, 0, d_src, size, cudaMemcpyDeviceToDevice));
	tex.normalized = 0;	// access without normalized texture coordinates
				// [0, width -1] [0, height - 1]
	// bind the array to the texture
	cutilSafeCall(cudaBindTextureToArray(tex, d_array, channelDesc));
}


extern "C" void meanShiftFilter(dim3 grid, dim3 threads, 
		cl_float4* d_luv, float offset,
		float width, float height,
		float sigmaS, float sigmaR,
		float rsigmaS, float rsigmaR)
{
	meanshiftfilter<<< grid, threads>>>(d_luv, offset, width, height, sigmaS, sigmaR, rsigmaS, rsigmaR);
}


#endif // #ifndef _MSFILTER_KERNEL_H_
