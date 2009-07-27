#include "filter.h"
#include "rgbluv.h"

#include <stdio.h>

extern float * h_src;
extern float * h_dst;

extern void connect(void);
extern void boundaries(void);

void filterGold();
void uniformLSearch(float, float, float*);

void computeGold(void)
{	
	filterGold();
	connect();
	boundaries();
}

void uniformSearchGold(float *Mh, float *yk, float* wsum)
{
	
	//Declare variables
	int	i, j;
	
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
			dataPoint = N * (i * width + j);
			data_l = h_src[dataPoint + 0];
			data_u = h_src[dataPoint + 1];
			data_v = h_src[dataPoint + 2];
			
			//Determine if inside search window
			//Calculate distance squared of sub-space s	
			dx = (j - yk[0]) / sigmaS;
			dy = (i - yk[1]) / sigmaS;
			dl = (data_l - yk[2]) / sigmaR;               
			du = (data_u - yk[3]) / sigmaR;               
			dv = (data_v - yk[4]) / sigmaR;               
			
			diff0 += dx * dx;
			diff0 += dy * dy;
			
#undef  BRANCH_FREE
			//#define BRANCH_FREE 1
#ifdef  BRANCH_FREE
			// Original statement: diff1 += 4 * dl * dl;
			// Der Wertebereich der Helligkeit liegt im 
			// Interval L* = 0 for black bis L* = 100 for white.
			diff1 += dl * dl;
			diff1 += (int)(yk[2]/80) * 3.0f * dl * dl;
#else
			if((yk[2] > 80)) 
				diff1 += 4.0f * dl * dl;
			else
				diff1 += dl * dl;
#endif
			
			diff1 += du * du;
			diff1 += dv * dv;
			
#ifdef  BRANCH_FREE
			//if its inside search window perform sum and count
			// For a uniform kernel weight == 1 for all feature points
			//int res = !((int)diff0 - (int)diff1);
			float mx = fmaxf(diff0, diff1);
			mx = floorf(mx);
			mx = !mx;
			
			// considered point is within sphere => accumulate to mean
			Mh[0] += j * mx;
			Mh[1] += i * mx;
			Mh[2] += data_l * mx;
			Mh[3] += data_u * mx;
			Mh[4] += data_v * mx;
			(*wsum) += 1 * mx; //weight
#else
			
			// If its inside search window perform sum and count
			// For a uniform kernel weight == 1 for all feature points
			if((diff0 < 1.0 && diff1 < 1.0))
			{
				
				// considered point is within sphere => accumulate to mean
				Mh[0] += j;
				Mh[1] += i;
				Mh[2] += data_l;
				Mh[3] += data_u;
				Mh[4] += data_v;
				(*wsum) += 1; //weight
			}
#endif			
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

void filterGold()
{
	// Declare Variables
	int   iterationCount, i;
	float mvAbs;
	
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
		
		yk[2] = h_src[N * i + 0]; // l
		yk[3] = h_src[N * i + 1]; // u
		yk[4] = h_src[N * i + 2]; // v
		
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
			latticeVectorGold(Mh, yk);
			
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
		
		//store result into msRawData...
		h_dst[N * i + 0] = (float)(yk[0 + 2]);
		h_dst[N * i + 1] = (float)(yk[1 + 2]);
		h_dst[N * i + 2] = (float)(yk[2 + 2]);
	}
	
	
	for(unsigned int i = 0; i < L; i++) {
		unsigned char * pix = (unsigned char *)&h_filt[i];
		LUVtoRGB(&h_dst[3 /*N*/ *i], pix);
	}
	
	// done.
	return;
}

