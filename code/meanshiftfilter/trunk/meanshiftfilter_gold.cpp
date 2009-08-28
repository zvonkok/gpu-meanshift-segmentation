#include "filter.h"
#include "rgbluv.h"

#include <iostream>
#include <cuda_runtime_api.h>


#include <sys/time.h>

#include <math.h>

#include "meanshiftfilter_common.h" 

struct timeval start, finish;
float msec;

extern float4 * h_src;
extern float4 * h_dst;
extern unsigned int * h_iter;


extern void connect(void);
extern void boundaries(void);

void filterGold();
void uniformLSearch(float, float, float*);


/* 
 CUDA Emulator Output
 
 As many people have noticed the same code executed in Emulator mode gives
 different floating point results from the kernels run in Debug or Release mode.
 Although I know what causes this I have never bothered to investigate the actual
 differences as most of the stuff I write runs entirely on the GPU. Recently I
 have had to compare results on the CPU<->GPU and wrote some code to change the
 FPU settings. Firstly a quick explanation: By default the CPU (FPU) is set to
 use 80 bit floating point internally. This means that when you load in an
 integer (fild) or a single / double float (fld) it gets converted to a 80 bit
 number inside the FPU stack. All operations are performed internally at 80 bits
 and when storing the result it converts back to the correct floating point width
 (single / double) (fst / fstp). This method of operation is desirable as it
 reduces the effect of rounding / truncating on the intermediate results. Of
 course while very useful for computing on the CPU this is not how the CUDA
 devices operate.
 
 In CUDA all operations on a float occur at 32 bits (64 bits for a double) which
 means your intermediate operations will sometimes lose precision. In CUDA
 Emulator mode your code is actually run on the CPU and it uses the FPU’s default
 precision and rounding settings. This causes the difference in output.
 
 For my testing I modified the Matrix Mul sample in the CUDA SDK to include code
 to change the CPU settings before running the Gold Kernel. (Code link follows
 below)
 
 I turned down the CPU internal precision to 32 bits in order to match the 32bit
 floats the CUDA kernel uses. For emulator mode I made sure the CPU was turned
 down to the same precision before running the Kernel. As expected the Gold and
 CUDA kernels outputs match perfectly.
 
 Next I ran in Debug mode (the kernel will now execute on the GPU). As both the
 Gold kernel and Cuda kernel are now at 32 bits I expected the results to be the
 same. Rather interestingly it turned out that they are slightly different. I
 then tried changing the CPU rounding settings hoping to get the results to match
 up.
 
 After trying all the rounding settings I discovered that the default setting
 (round to nearest or even) gave the closest results to the Gold kernel BUT they
 are still slightly out. I suspect this is down to differences in the internal
 workings of the FPU units on the GPU.
 
 So in summary: If you are trying to compare kernel results between Emulator and
 Release mode you will never get exactly the same results but the differences can
 be mitigated somewhat by changing the CPU/FPU’s internal precision settings.
 */
void computeGold(void)
{	
	// set the precision and round to match CUDA hardware
	// precision can be 24, 53 ,64  ->  32, 64, 80  :  defaults to 80
	// rounding can be:
	//    0 : truncating mode
	//    1 : round to nearest or even      : default
	//    2 : round up
	//    3 : round down
	/*
	typedef unsigned char BYTE;
	typedef unsigned short WORD;
	extern void set_FPU_Precision_Rounding(BYTE precision, BYTE rounding);
	set_FPU_Precision_Rounding(53, 0);
	*/
	
	
	
#ifdef __linux__
	gettimeofday(&start, NULL);
	filterGold();
	gettimeofday(&finish, NULL);
	
	msec = finish.tv_sec * 1000.0f + finish.tv_usec / 1000.0f;
	msec -= start.tv_sec * 1000.0f + start.tv_usec / 1000.0f;
	
	printf("Processing time (GOLD): %f (ms) \n", msec);
#else
	filterGold();
#endif	
	connect();
	boundaries();
}

void uniformSearchGold(float *Mh, float *yk, float* wsum)
{
	
	//Declare variables
	int	i, j, h = height, w = width;
	
	int	dataPoint;
	
	float diff0, diff1;
	float dx, dy, dl, du, dv;
	
	float data_l, data_u, data_v;
	//Define bounds of lattice...
	
	//the lattice is a 2dimensional subspace whose
	//search window bandwidth is specified by sigmaS:
	int LowerBoundX = yk[0] - sigmaS;
	int LowerBoundY = yk[1] - sigmaS;
	int UpperBoundX = yk[0] + sigmaS;
	int UpperBoundY = yk[1] + sigmaS;
	
	if (LowerBoundX < 0)
		LowerBoundX = 0;
	if (LowerBoundY < 0)
		LowerBoundY = 0;
	if (UpperBoundX >= w)
		UpperBoundX = w - 1;
	if (UpperBoundY >= h)
		UpperBoundY = h - 1;
	
	//Perform search using lattice
	//Iterate once through a window of size sigmaS
	for(i = LowerBoundY; i <= UpperBoundY; i++)
		for(j = LowerBoundX; j <= UpperBoundX; j++)
		{
			diff0 = 0;
			diff1 = 0;
			
			//get index into data array
			dataPoint = (i * width + j);
			
			data_l = h_src[dataPoint].x;
			data_u = h_src[dataPoint].y;
			data_v = h_src[dataPoint].z;
			
			//Determine if inside search window
			//Calculate distance squared of sub-space s	
			dx = (j - yk[0]) / sigmaS;
			dy = (i - yk[1]) / sigmaS;
			dl = (data_l - yk[2]) / sigmaR;               
			du = (data_u - yk[3]) / sigmaR;               
			dv = (data_v - yk[4]) / sigmaR;               
			
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
				Mh[2] += data_l;
				Mh[3] += data_u;
				Mh[4] += data_v;
				(*wsum) += 1.0f; //weight
			}
	
		}
	
	return;
}


void latticeVectorGold(float *Mh_ptr, float *yk_ptr)
{
	
	// Initialize mean shift vector
	
	Mh_ptr[0] = 0;
	Mh_ptr[1] = 0;
	Mh_ptr[2] = 0;
	Mh_ptr[3] = 0;
	Mh_ptr[4] = 0;
	
	// Initialize wsum
	float wsum = 0;
	
	// Perform lattice search summing
	// all the points that lie within the search
	// window defined using the kernel specified
	// by uniformKernel
	uniformSearchGold(Mh_ptr, yk_ptr, &wsum);
	
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

void getColor( int iter, unsigned char* pix) 
{
	if (iter >= 0 && iter < 10) {
		pix[0] = 29;
		pix[1] = 29;
		pix[2] = 29;
		return;
	}
	if (iter >= 10 && iter < 20) {
		pix[0] = 58;
		pix[1] = 58;
		pix[2] = 58;

		return;
		
	}
	if (iter >= 20 && iter < 30) {
		pix[0] = 130;
		pix[1] = 106;
		pix[2] = 129;
		return;
	}
	if (iter >= 30 && iter < 40) {
		pix[0] = 92;
		pix[1] = 110;
		pix[2] = 137;
		return;
	}
	
	if (iter >= 40 && iter < 50) {
		pix[0] = 136;
		pix[1] = 156;
		pix[2] = 176;
		return;
	}
	if (iter >= 50 && iter < 60) {
		pix[0] = 152;
		pix[1] = 157;
		pix[2] = 118;
		return;
	}
	if (iter >= 60 && iter < 70) {
		pix[0] = 117;
		pix[1] = 128;
		pix[2] = 88;
		return;
	}
	if (iter >= 70 && iter < 80) {
		pix[0] = 201;
		pix[1] = 193;
		pix[2] = 114;
		return;
	}
	if (iter >= 80) {
		pix[0] = 181;
		pix[1] = 80;
		pix[2] = 54;
		return;
	}

}

void filterGold()
{
	// Declare Variables
	int   iter;
	volatile float mvAbs;
	
	// Traverse each data point applying mean shift
	// to each data point
	
	// Allcocate memory for yk
	float	yk[5];
	float	Mh[5];
	
	for(unsigned int i = 0; i < L; i++)
	{
			
		// Assign window center (window centers are
		// initialized by createLattice to be the point
		// data[i])
		yk[0] = (float)(i%width); // x
		yk[1] = (float)(i/width); // y 
		
		yk[2] = h_src[i].x; // l
		yk[3] = h_src[i].y; // u
		yk[4] = h_src[i].z; // v
		
		// Calculate the mean shift vector using the lattice
		latticeVectorGold(Mh, yk);
		
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
		iter = 1;
		
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

		while((mvAbs >= EPSILON) && (iter < LIMIT))
		{
			// Shift window location
			yk[0] += Mh[0];
			yk[1] += Mh[1];
			yk[2] += Mh[2];
			yk[3] += Mh[3];
			yk[4] += Mh[4];
			
			
			
			// Calculate the mean shift vector at the new
			// window location using lattice
			latticeVectorGold(Mh, yk);
			
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
			
		}
		
		// Shift window location
		yk[0] += Mh[0];
		yk[1] += Mh[1];
		yk[2] += Mh[2];
		yk[3] += Mh[3];
		yk[4] += Mh[4];
		
		
		//store result into msRawData...
		h_dst[i].x = (float)(yk[0 + 2]);
		h_dst[i].y = (float)(yk[1 + 2]);
		h_dst[i].z = (float)(yk[2 + 2]);
	
		
		
#if 0	
		// store iteration count for each pixel
		unsigned char* pix = (unsigned char *)&h_iter[i];
		getColor(iter, pix);
#endif		
	
	}
	
	
	for(unsigned int i = 0; i < L; i++) {
		unsigned char * pix = (unsigned char *)&h_filt[i];
		LUVtoRGB((float*)&h_dst[i], pix);
	}
	
	// done.
	return;
}
