////////////////////////////////////////////////////////
// Name     : edison.cpp
// Purpose  : Wrapper class used for segmenation and
//            edge detection.
// Author   : Chris M. Christoudias
// Modified by
// Created  : 03/20/2002
// Copyright: (c) Chris M. Christoudias
// Version  : v0.1
////////////////////////////////////////////////////////
#define _CRT_SECURE_NO_WARNINGS 1

#include "rlist.h"
#include "rgbluv.h"
#include "filter.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

#include <GL/glew.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_runtime_api.h>
#include <cutil_inline.h>

//////////////////////////////////////////////////////////////////////////////////////
// cleaned up stuff 
float sigmaS;
float sigmaR;
float rcpr_sigmaS;
float rcpr_sigmaR;

float minRegion;



extern unsigned int height;
extern unsigned int width;	

extern unsigned int * h_filt;
extern unsigned int * h_segm;
extern unsigned char * h_bndy;

extern float4 * h_src;
extern float4 * h_dst;



int *boundaries_;
int numBoundaries_;

unsigned int L;
unsigned int N;


int neigh[8];			// Connect
int *labels;			// Connect
int	*modePointCounts;	// Connect
float *modes;			// Connect

int	*indexTable;			// Fill
float LUV_treshold = 1.0f;	// Fill
unsigned int regionCount;	// Fill

RegionList *regionList;		// DefineBoundaries

unsigned char *visitTable;	// FuseRegions
float rR2;					// FuseRegions	


RAList *raList;				// BuildRam
RAList *freeRAList;			// BuildRam
RAList *raPool;				// BuildRam	

float *weightMap;			// TransitiveClosure
float epsilon = 1.0f;		// TransitiveClosure
//////////////////////////////////////////////////////////////////////////////////////


void Fill(int regionLoc, int label)
{
	//declare variables
	unsigned int k;
	int neighLoc, neighborsFound, imageSize	= width*height;
	
	//Fill region starting at region location
	//using labels...
	
	//initialzie indexTable
	int	index		= 0;
	indexTable[0]	= regionLoc;
	
	//increment mode point counts for this region to
	//indicate that one pixel belongs to this region
	modePointCounts[label]++;
	
	while(true)
	{
		
		//assume no neighbors will be found
		neighborsFound	= 0;
		
		//check the eight connected neighbors at regionLoc -
		//if a pixel has similar color to that located at 
		//regionLoc then declare it as part of this region
		for (unsigned int i = 0; i < 8; i++)
		{
			//check bounds and if neighbor has been already labeled
			neighLoc			= regionLoc + neigh[i];
			if ((neighLoc >= 0)&&(neighLoc < imageSize)&&(labels[neighLoc] < 0))
			{
				for (k = 0; k < N; k++)
				{
					if (fabs(h_dst[regionLoc].x - h_dst[neighLoc].x) >= LUV_treshold)
						break;
					if (fabs(h_dst[regionLoc].y - h_dst[neighLoc].y) >= LUV_treshold)
						break;
					if (fabs(h_dst[regionLoc].z - h_dst[neighLoc].z) >= LUV_treshold)
						break;
				}
				
				//neighbor i belongs to this region so label it and
				//place it onto the index table buffer for further
				//processing
				if (k == N)
				{
					//assign label to neighbor i
					labels[neighLoc]	= label;
					
					//increment region point count
					modePointCounts[label]++;
					
					//place index of neighbor i onto the index tabel buffer
					indexTable[++index]	= neighLoc;
					
					//indicate that a neighboring region pixel was
					//identified
					neighborsFound	= 1;
				}
			}
		}
		
		//check the indexTable to see if there are any more
		//entries to be explored - if so explore them, otherwise
		//exit the loop - we are finished
		if(neighborsFound)
			regionLoc	= indexTable[index];
		else if (index > 1)
			regionLoc	= indexTable[--index];
		else
			break; //fill complete
	}
	
	//done.
	return;
	
}

void Connect( void )
{
	//define eight connected neighbors
	neigh[0]	= 1;
	neigh[1]	= 1-width;
	neigh[2]	= -width;
	neigh[3]	= -(1+width);
	neigh[4]	= -1;
	neigh[5]	= width-1;
	neigh[6]	= width;
	neigh[7]	= width+1;
	
	//initialize labels and modePointCounts
	unsigned int i;
	for(i = 0; i < width*height; i++)
	{
		labels[i]			= -1;
		modePointCounts[i]	=  0;
	}
	
	//Traverse the image labeling each new region encountered
	unsigned int label = -1;
	for(i = 0; i < height*width; i++)
	{
		//if this region has not yet been labeled - label it
		if(labels[i] < 0)
		{
			//assign new label to this region
			labels[i] = ++label;
			
			//copy region color into modes
			modes[(N*label)+0] = h_dst[i].x;
			modes[(N*label)+1] = h_dst[i].y;
			modes[(N*label)+2] = h_dst[i].z;
			
			//populate labels with label for this specified region
			//calculating modePointCounts[label]...
			Fill(i, label);
		}
	}
	
	//calculate region count using label
	regionCount	= label+1;
	
	//done.
	return;
}

void DefineBoundaries( void )
{
	//declare and allocate memory for boundary map and count
	int	*boundaryMap,	*boundaryCount = NULL;
	if((!(boundaryMap = new int [L]))||(!(boundaryCount = new int [regionCount])));
	
	//initialize boundary map and count
	unsigned int i;
	for(i = 0; i < L; i++)
		boundaryMap[i]		= -1;
	for(i = 0; i < regionCount; i++)
		boundaryCount[i]	=  0;
	
	//initialize and declare total boundary count -
	//the total number of boundary pixels present in
	//the segmented image
	int	totalBoundaryCount	= 0;
	
	//traverse the image checking the right and bottom
	//four connected neighbors of each pixel marking
	//boundary map with the boundaries of each region and
	//incrementing boundaryCount using the label information
	
	//***********************************************************************
	//***********************************************************************
	
	int	label, dataPoint;
	
	//first row (every pixel is a boundary pixel)
	for(i = 0; i < width; i++)
	{
		boundaryMap[i]		= label	= labels[i];
		boundaryCount[label]++;
		totalBoundaryCount++;
	}
	
	//define boundaries for all rows except for the first
	//and last one...
	for(i = 1; i < height - 1; i++)
	{
		//mark the first pixel in an image row as an image boundary...
		dataPoint				= i*width;
		boundaryMap[dataPoint]	= label	= labels[dataPoint];
		boundaryCount[label]++;
		totalBoundaryCount++;
		
		for(unsigned int j = 1; j < width - 1; j++)
		{
			//define datapoint and its right and bottom
			//four connected neighbors
			dataPoint		= i*width+j;
			
			//check four connected neighbors if they are
			//different this pixel is a boundary pixel
			label	= labels[dataPoint];
			if((label != labels[dataPoint-1])    ||(label != labels[dataPoint+1])||
			   (label != labels[dataPoint-width])||(label != labels[dataPoint+width]))
			{
				boundaryMap[dataPoint]		= label	= labels[dataPoint];
				boundaryCount[label]++;
				totalBoundaryCount++;
			}
		}
		
		//mark the last pixel in an image row as an image boundary...
		dataPoint				= (i+1)*width-1;
		boundaryMap[dataPoint]	= label	= labels[dataPoint];
		boundaryCount[label]++;
		totalBoundaryCount++;
		
	}
	
	//last row (every pixel is a boundary pixel) (i = height-1)
	unsigned int start	= (height-1)*width, stop = height*width;
	for(i = start; i < stop; i++)
	{
		boundaryMap[i]		= label	= labels[i];
		boundaryCount[label]++;
		totalBoundaryCount++;
	}
	
	//***********************************************************************
	//***********************************************************************
	
	//store boundary locations into a boundary buffer using
	//boundary map and count
	
	//***********************************************************************
	//***********************************************************************
	
	int	*boundaryBuffer	= new int [totalBoundaryCount], *boundaryIndex	= new int [regionCount];
	
	//use boundary count to initialize boundary index...
	int counter = 0;
	for(i = 0; i < regionCount; i++)
	{
		boundaryIndex[i]	= counter;
		counter			   += boundaryCount[i];
	}
	
	//traverse boundary map placing the boundary pixel
	//locations into the boundaryBuffer
	for(i = 0; i < L; i++)
	{
		//if its a boundary pixel store it into
		//the boundary buffer
		if((label = boundaryMap[i]) >= 0)
		{
			boundaryBuffer[boundaryIndex[label]] = i;
			boundaryIndex[label]++;
		}
	}
	
	//***********************************************************************
	//***********************************************************************
	
	//store the boundary locations stored by boundaryBuffer into
	//the region list for each region
	
	//***********************************************************************
	//***********************************************************************
	
	//destroy the old region list
	if(regionList)	delete regionList;
	
	//create a new region list
	if(!(regionList	= new RegionList(regionCount, totalBoundaryCount, N)))
		;//ErrorHandler("msImageProcessor", "DefineBoundaries", "Not enough memory.");
	
	//add boundary locations for each region using the boundary
	//buffer and boundary counts
	counter	= 0;
	for(i = 0; i < regionCount; i++)
	{
		regionList->AddRegion(i, boundaryCount[i], &boundaryBuffer[counter]);
		counter += boundaryCount[i];
	}
	
	//***********************************************************************
	//***********************************************************************
	
	// dealocate local used memory
	delete [] boundaryMap;
	delete [] boundaryCount;
	delete [] boundaryBuffer;
	delete [] boundaryIndex;
	
	//done.
	return;
	
}
RegionList *GetBoundaries( void )
{
	DefineBoundaries();
	return regionList;
}
bool InWindow(int mode1, int mode2)
{
	// ISRUN
	int		k		= 1, s	= 0, p;
	double	diff	= 0, el;
	while((diff < 0.25)&&(k != 2)) // Partial Distortion Search
	{
		//Calculate distance squared of sub-space s	
		diff = 0;
		for(p = 0; p < 3; p++)
		{
			el    = (modes[mode1*N+p+s]-modes[mode2*N+p+s])/(sigmaR);
			if((!p)&&(k == 1)&&(modes[mode1*N] > 80))
				diff += 4*el*el;
			else
				diff += el*el;
		}
		
		//next subspace
		s += 3;
		k++;
	}
	return (bool)(diff < 0.25);
}

float SqDistance(int mode1, int mode2)
{
	
	// ISRUN
	int		k		= 1, s	= 0, p;
	float	dist	= 0, el;
	for(k = 1; k < 2; k++)
	{
		//Calculate distance squared of sub-space s	
		for(p = 0; p < 3; p++)
		{
			el    = (modes[mode1*N+p+s]-modes[mode2*N+p+s])/(sigmaR);
			dist += el*el;
		}
		
		//next subspace
		s += 3;
		k++;
	}
	
	//return normalized square distance between modes
	//1 and 2
	return dist;
	
}

#define NODE_MULTIPLE 10

void BuildRAM( void )
{
	//Allocate memory for region adjacency matrix if it hasn't already been allocated
	if((!raList)&&((!(raList = new RAList [regionCount]))||(!(raPool = new RAList [NODE_MULTIPLE*regionCount]))))
	{
		//ErrorHandler("msImageProcessor", "Allocate", "Not enough memory.");
		return;
	}
	
	//initialize the region adjacency list
	unsigned int i;
	for(i = 0; i < regionCount; i++)
	{
		raList[i].edgeStrength		= 0;
		raList[i].edgePixelCount	= 0;
		raList[i].label				= i;
		raList[i].next				= NULL;
	}
	
	//initialize RAM free list
	freeRAList	= raPool;
	for(i = 0; i < NODE_MULTIPLE*regionCount-1; i++)
	{
		raPool[i].edgeStrength		= 0;
		raPool[i].edgePixelCount	= 0;
		raPool[i].next = &raPool[i+1];
	}
	raPool[NODE_MULTIPLE*regionCount-1].next	= NULL;
	
	//traverse the labeled image building
	//the RAM by looking to the right of
	//and below the current pixel location thus
	//determining if a given region is adjacent
	//to another
	unsigned int j;
	int	curLabel, rightLabel, bottomLabel, exists;
	RAList	*raNode1, *raNode2, *oldRAFreeList;
	for(i = 0; i < height - 1; i++)
	{
		//check the right and below neighbors
		//for pixel locations whose x < width - 1
		for(j = 0; j < width - 1; j++)
		{
			//calculate pixel labels
			curLabel	= labels[i*width+j    ];	//current pixel
			rightLabel	= labels[i*width+j+1  ];	//right   pixel
			bottomLabel	= labels[(i+1)*width+j];	//bottom  pixel
			
			//check to the right, if the label of
			//the right pixel is not the same as that
			//of the current one then region[j] and region[j+1]
			//are adjacent to one another - update the RAM
			if(curLabel != rightLabel)
			{
				//obtain RAList object from region adjacency free
				//list
				raNode1			= freeRAList;
				raNode2			= freeRAList->next;
				
				//keep a pointer to the old region adj. free
				//list just in case nodes already exist in respective
				//region lists
				oldRAFreeList	= freeRAList;
				
				//update region adjacency free list
				freeRAList		= freeRAList->next->next;
				
				//populate RAList nodes
				raNode1->label	= curLabel;
				raNode2->label	= rightLabel;
				
				//insert nodes into the RAM
				exists			= 0;
				raList[curLabel  ].Insert(raNode2);
				exists			= raList[rightLabel].Insert(raNode1);
				
				//if the node already exists then place
				//nodes back onto the region adjacency
				//free list
				if(exists)
					freeRAList = oldRAFreeList;
				
			}
			
			//check below, if the label of
			//the bottom pixel is not the same as that
			//of the current one then region[j] and region[j+width]
			//are adjacent to one another - update the RAM
			if(curLabel != bottomLabel)
			{
				//obtain RAList object from region adjacency free
				//list
				raNode1			= freeRAList;
				raNode2			= freeRAList->next;
				
				//keep a pointer to the old region adj. free
				//list just in case nodes already exist in respective
				//region lists
				oldRAFreeList	= freeRAList;
				
				//update region adjacency free list
				freeRAList		= freeRAList->next->next;
				
				//populate RAList nodes
				raNode1->label	= curLabel;
				raNode2->label	= bottomLabel;
				
				//insert nodes into the RAM
				exists			= 0;
				raList[curLabel  ].Insert(raNode2);
				exists			= raList[bottomLabel].Insert(raNode1);
				
				//if the node already exists then place
				//nodes back onto the region adjacency
				//free list
				if(exists)
					freeRAList = oldRAFreeList;
				
			}
			
		}
		
		//check only to the bottom neighbors of the right boundary
		//pixels...
		
		//calculate pixel locations (j = width-1)
		curLabel	= labels[i*width+j    ];	//current pixel
		bottomLabel = labels[(i+1)*width+j];	//bottom  pixel
		
		//check below, if the label of
		//the bottom pixel is not the same as that
		//of the current one then region[j] and region[j+width]
		//are adjacent to one another - update the RAM
		if(curLabel != bottomLabel)
		{
			//obtain RAList object from region adjacency free
			//list
			raNode1			= freeRAList;
			raNode2			= freeRAList->next;
			
			//keep a pointer to the old region adj. free
			//list just in case nodes already exist in respective
			//region lists
			oldRAFreeList	= freeRAList;
			
			//update region adjacency free list
			freeRAList		= freeRAList->next->next;
			
			//populate RAList nodes
			raNode1->label	= curLabel;
			raNode2->label	= bottomLabel;
			
			//insert nodes into the RAM
			exists			= 0;
			raList[curLabel  ].Insert(raNode2);
			exists			= raList[bottomLabel].Insert(raNode1);
			
			//if the node already exists then place
			//nodes back onto the region adjacency
			//free list
			if(exists)
				freeRAList = oldRAFreeList;
			
		}
	}
	
	//check only to the right neighbors of the bottom boundary
	//pixels...
	
	//check the right for pixel locations whose x < width - 1
	for(j = 0; j < width - 1; j++)
	{
		//calculate pixel labels (i = height-1)
		curLabel	= labels[i*width+j    ];	//current pixel
		rightLabel	= labels[i*width+j+1  ];	//right   pixel
		
		//check to the right, if the label of
		//the right pixel is not the same as that
		//of the current one then region[j] and region[j+1]
		//are adjacent to one another - update the RAM
		if(curLabel != rightLabel)
		{
			//obtain RAList object from region adjacency free
			//list
			raNode1			= freeRAList;
			raNode2			= freeRAList->next;
			
			//keep a pointer to the old region adj. free
			//list just in case nodes already exist in respective
			//region lists
			oldRAFreeList	= freeRAList;
			
			//update region adjacency free list
			freeRAList		= freeRAList->next->next;
			
			//populate RAList nodes
			raNode1->label	= curLabel;
			raNode2->label	= rightLabel;
			
			//insert nodes into the RAM
			exists			= 0;
			raList[curLabel  ].Insert(raNode2);
			exists			= raList[rightLabel].Insert(raNode1);
			
			//if the node already exists then place
			//nodes back onto the region adjacency
			//free list
			if(exists)
				freeRAList = oldRAFreeList;
			
		}
		
	}
	
	//done.
	return;
	
}


void Prune(int minRegion)
{
	
	//allocate memory for mode and point count temporary buffers...
	float	*modes_buffer	= new float	[N*regionCount];
	int		*MPC_buffer		= new int	[regionCount];
	
	//allocate memory for label buffer
	int	*label_buffer		= new int	[regionCount];
	
	//Declare variables
	int candidate, iCanEl, neighCanEl, iMPC, label, oldRegionCount, minRegionCount;
	double	minSqDistance, neighborDistance;
	RAList	*neighbor;
	
	//Apply pruning algorithm to classification structure, removing all regions whose area
	//is under the threshold area minRegion (pixels)
	do
	{
		//Assume that no region has area under threshold area  of 
		minRegionCount	= 0;		
		
		//Step (1):
		
		// Build RAM using classifiction structure originally
		// generated by the method GridTable::Connect()
		BuildRAM();
		
		// Step (2):
		
		// Traverse the RAM joining regions whose area is less than minRegion (pixels)
		// with its respective candidate region.
		
		// A candidate region is a region that displays the following properties:
		
		//	- it is adjacent to the region being pruned
		
		//  - the distance of its mode is a minimum to that of the region being pruned
		//    such that or it is the only adjacent region having an area greater than
		//    minRegion
		
		for(unsigned int i = 0; i < regionCount; i++)
		{
			//if the area of the ith region is less than minRegion
			//join it with its candidate region...
			
			//*******************************************************************************
			
			//Note: Adjust this if statement if a more sophisticated pruning criterion
			//      is desired. Basically in this step a region whose area is less than
			//      minRegion is pruned by joining it with its "closest" neighbor (in color).
			//      Therefore, by placing a different criterion for fusing a region the
			//      pruning method may be altered to implement a more sophisticated algorithm.
			
			//*******************************************************************************
			
			if(modePointCounts[i] < minRegion)
			{
				//update minRegionCount to indicate that a region
				//having area less than minRegion was found
				minRegionCount++;
				
				//obtain a pointer to the first region in the
				//region adjacency list of the ith region...
				neighbor	= raList[i].next;
				
				//calculate the distance between the mode of the ith
				//region and that of the neighboring region...
				candidate		= neighbor->label;
				minSqDistance	= SqDistance(i, candidate);
				
				//traverse region adjacency list of region i and select
				//a candidate region
				neighbor	= neighbor->next;
				while(neighbor)
				{
					
					//calculate the square distance between region i
					//and current neighbor...
					neighborDistance = SqDistance(i, neighbor->label);
					
					//if this neighbors square distance to region i is less
					//than minSqDistance, then select this neighbor as the
					//candidate region for region i
					if(neighborDistance < minSqDistance)
					{
						minSqDistance	= neighborDistance;
						candidate		= neighbor->label;
					}
					
					//traverse region list of region i
					neighbor	= neighbor->next;
					
				}
				
				//join region i with its candidate region:
				
				// (1) find the canonical element of region i
				iCanEl		= i;
				while(raList[iCanEl].label != iCanEl)
					iCanEl		= raList[iCanEl].label;
				
				// (2) find the canonical element of neighboring region
				neighCanEl	= candidate;
				while(raList[neighCanEl].label != neighCanEl)
					neighCanEl	= raList[neighCanEl].label;
				
				// if the canonical elements of are not the same then assign
				// the canonical element having the smaller label to be the parent
				// of the other region...
				if(iCanEl < neighCanEl)
					raList[neighCanEl].label	= iCanEl;
				else
				{
					//must replace the canonical element of previous
					//parent as well
					raList[raList[iCanEl].label].label	= neighCanEl;
					
					//re-assign canonical element
					raList[iCanEl].label				= neighCanEl;
				}
			}
		}
		
		// Step (3):
		
		// Level binary trees formed by canonical elements
		for(unsigned int i = 0; i < regionCount; i++)
		{
			iCanEl	= i;
			while(raList[iCanEl].label != iCanEl)
				iCanEl	= raList[iCanEl].label;
			raList[i].label	= iCanEl;
		}
		
		// Step (4):
		
		//Traverse joint sets, relabeling image.
		
		// Accumulate modes and re-compute point counts using canonical
		// elements generated by step 2.
		
		//initialize buffers to zero
		for(unsigned int i = 0; i < regionCount; i++)
			MPC_buffer[i]	= 0;
		for(unsigned int i = 0; i < N*regionCount; i++)
			modes_buffer[i]	= 0;
		
		//traverse raList accumulating modes and point counts
		//using canoncial element information...
		for(unsigned int i = 0; i < regionCount; i++)
		{
			
			//obtain canonical element of region i
			iCanEl	= raList[i].label;
			
			//obtain mode point count of region i
			iMPC	= modePointCounts[i];
			
			//accumulate modes_buffer[iCanEl]
			for(unsigned int k = 0; k < N; k++)
				modes_buffer[(N*iCanEl)+k] += iMPC*modes[(N*i)+k];
			
			//accumulate MPC_buffer[iCanEl]
			MPC_buffer[iCanEl] += iMPC;
			
		}
		
		// (b)
		
		// Re-label new regions of the image using the canonical
		// element information generated by step (2)
		
		// Also use this information to compute the modes of the newly
		// defined regions, and to assign new region point counts in
		// a consecute manner to the modePointCounts array
		
		//initialize label buffer to -1
		for(unsigned int i = 0; i < regionCount; i++)
			label_buffer[i]	= -1;
		
		//traverse raList re-labeling the regions
		label = -1;
		for(unsigned int i = 0; i < regionCount; i++)
		{
			//obtain canonical element of region i
			iCanEl	= raList[i].label;
			if(label_buffer[iCanEl] < 0)
			{
				//assign a label to the new region indicated by canonical
				//element of i
				label_buffer[iCanEl]	= ++label;
				
				//recompute mode storing the result in modes[label]...
				iMPC	= MPC_buffer[iCanEl];
				for(unsigned int k = 0; k < N; k++)
					modes[(N*label)+k]	= (modes_buffer[(N*iCanEl)+k])/(iMPC);
				
				//assign a corresponding mode point count for this region into
				//the mode point counts array using the MPC buffer...
				modePointCounts[label]	= MPC_buffer[iCanEl];
			}
		}
		
		//re-assign region count using label counter
		oldRegionCount	= regionCount;
		regionCount		= label+1;
		
		// (c)
		
		// Use the label buffer to reconstruct the label map, which specified
		// the new image given its new regions calculated above
		
		for(unsigned int i = 0; i < height*width; i++)
			labels[i]	= label_buffer[raList[labels[i]].label];
		
		
	}	while(minRegionCount > 0);
	
	//de-allocate memory
	delete [] modes_buffer;
	delete [] MPC_buffer;
	delete [] label_buffer;
	
	//done.
	return;
	
}


void TransitiveClosure(void)
{
	// Build RAM using classifiction structure originally
	// generated by the method GridTable::Connect()
	BuildRAM();
	
	//Step (1a):
	//Compute weights of weight graph using confidence map
	//(if defined)
	// NOT USED ... if(weightMapDefined)	ComputeEdgeStrengths();
	
	//Step (2):
	
	//Treat each region Ri as a disjoint set:
	
	// - attempt to join Ri and Rj for all i != j that are neighbors and
	//   whose associated modes are a normalized distance of < 0.5 from one
	//   another
	
	// - the label of each region in the raList is treated as a pointer to the
	//   canonical element of that region (e.g. raList[i], initially has raList[i].label = i,
	//   namely each region is initialized to have itself as its canonical element).
	
	//Traverse RAM attempting to join raList[i] with its neighbors...
	int	iCanEl, neighCanEl;
	float	threshold;
	RAList	*neighbor;
	for(unsigned int i = 0; i < regionCount; i++)
	{
		//aquire first neighbor in region adjacency list pointed to
		//by raList[i]
		neighbor	= raList[i].next;
		
		//compute edge strenght threshold using global and local
		//epsilon
		if(epsilon > raList[i].edgeStrength)
			threshold   = epsilon;
		else
			threshold   = raList[i].edgeStrength;
		
		//traverse region adjacency list of region i, attempting to join
		//it with regions whose mode is a normalized distance < 0.5 from
		//that of region i...
		while(neighbor)
		{
			//attempt to join region and neighbor...
			if((InWindow(i, neighbor->label))&&(neighbor->edgeStrength < epsilon))
			{
				//region i and neighbor belong together so join them
				//by:
				
				// (1) find the canonical element of region i
				iCanEl		= i;
				while(raList[iCanEl].label != iCanEl)
					iCanEl		= raList[iCanEl].label;
				
				// (2) find the canonical element of neighboring region
				neighCanEl	= neighbor->label;
				while(raList[neighCanEl].label != neighCanEl)
					neighCanEl	= raList[neighCanEl].label;
				
				// if the canonical elements of are not the same then assign
				// the canonical element having the smaller label to be the parent
				// of the other region...
				if(iCanEl < neighCanEl)
					raList[neighCanEl].label	= iCanEl;
				else
				{
					//must replace the canonical element of previous
					//parent as well
					raList[raList[iCanEl].label].label	= neighCanEl;
					
					//re-assign canonical element
					raList[iCanEl].label				= neighCanEl;
				}
			}
			
			//check the next neighbor...
			neighbor	= neighbor->next;
			
		}
	}
	
	// Step (3):
	
	// Level binary trees formed by canonical elements
	for(unsigned int i = 0; i < regionCount; i++)
	{
		iCanEl	= i;
		while(raList[iCanEl].label != iCanEl)
			iCanEl	= raList[iCanEl].label;
		raList[i].label	= iCanEl;
	}
	
	// Step (4):
	
	//Traverse joint sets, relabeling image.
	
	// (a)
	
	// Accumulate modes and re-compute point counts using canonical
	// elements generated by step 2.
	
	//allocate memory for mode and point count temporary buffers...
	float	*modes_buffer	= new float	[N*regionCount];
	int		*MPC_buffer		= new int	[regionCount];
	
	//initialize buffers to zero
	for(unsigned int i = 0; i < regionCount; i++)
		MPC_buffer[i]	= 0;
	for(unsigned int i = 0; i < N*regionCount; i++)
		modes_buffer[i]	= 0;
	
	//traverse raList accumulating modes and point counts
	//using canoncial element information...
	int iMPC;
	for(unsigned int i = 0; i < regionCount; i++) {
		//obtain canonical element of region i
		iCanEl	= raList[i].label;
		
		//obtain mode point count of region i
		iMPC	= modePointCounts[i];
		
		//accumulate modes_buffer[iCanEl]
		for(unsigned int k = 0; k < N; k++)
			modes_buffer[(N*iCanEl)+k] += iMPC*modes[(N*i)+k];
		
		//accumulate MPC_buffer[iCanEl]
		MPC_buffer[iCanEl] += iMPC;
		
	}
	
	// (b)
	
	// Re-label new regions of the image using the canonical
	// element information generated by step (2)
	
	// Also use this information to compute the modes of the newly
	// defined regions, and to assign new region point counts in
	// a consecute manner to the modePointCounts array
	
	//allocate memory for label buffer
	int	*label_buffer	= new int [regionCount];
	
	//initialize label buffer to -1
	for(unsigned int i = 0; i < regionCount; i++)
		label_buffer[i]	= -1;
	
	//traverse raList re-labeling the regions
	int	label = -1;
	for(unsigned int i = 0; i < regionCount; i++)
	{
		//obtain canonical element of region i
		iCanEl	= raList[i].label;
		if(label_buffer[iCanEl] < 0)
		{
			//assign a label to the new region indicated by canonical
			//element of i
			label_buffer[iCanEl]	= ++label;
			
			//recompute mode storing the result in modes[label]...
			iMPC	= MPC_buffer[iCanEl];
			for(unsigned int k = 0; k < N; k++)
				modes[(N*label)+k]	= (modes_buffer[(N*iCanEl)+k])/(iMPC);
			
			//assign a corresponding mode point count for this region into
			//the mode point counts array using the MPC buffer...
			modePointCounts[label]	= MPC_buffer[iCanEl];
		}
	}
	
	//re-assign region count using label counter
	// int	oldRegionCount	= regionCount;
	regionCount	= label+1;
	
	// (c)
	
	// Use the label buffer to reconstruct the label map, which specified
	// the new image given its new regions calculated above
	
	for(unsigned int i = 0; i < height*width; i++)
		labels[i]	= label_buffer[raList[labels[i]].label];
	
	//de-allocate memory
	delete [] modes_buffer;
	delete [] MPC_buffer;
	delete [] label_buffer;
	
	//done.
	return;
	
}

void DestroyRAM( void )
{
	//de-allocate memory for region adjaceny list
	if (raList)				delete [] raList;
	if (raPool)				delete [] raPool;
	
	//initialize region adjacency matrix
	raList				= NULL;
	freeRAList			= NULL;
	raPool				= NULL;
	
	//done.
	return;
	
}


void FuseRegions(float sigmaS, int minRegion)
{
	//allocate memory visit table
	visitTable = new unsigned char [L];
	
	//Apply transitive closure iteratively to the regions classified
	//by the RAM updating labels and modes until the color of each neighboring
	//region is within sqrt(rR2) of one another.
	rR2 = (float)(sigmaS*sigmaS*0.25);
	TransitiveClosure();
	int oldRC = regionCount;
	int deltaRC, counter = 0;
	do {
		TransitiveClosure();
		deltaRC = oldRC-regionCount;
		oldRC = regionCount;
		counter++;
	} while ((deltaRC <= 0)&&(counter < 10));
	
	//de-allocate memory for visit table
	delete [] visitTable;
	visitTable	= NULL;
	
	//Prune spurious regions (regions whose area is under
	//minRegion) using RAM
	Prune(minRegion);
	
	//de-allocate memory for region adjacency matrix
	DestroyRAM();
	
	//output to h_dst
	int label;
	for(unsigned int i = 0; i < L; i++)
	{
		label	= labels[i];
		h_dst[i].x = modes[N*label+0];
		h_dst[i].y = modes[N*label+1];
		h_dst[i].z = modes[N*label+2];
	}
	
	//done.
	return;
	
}

void connect() 
{
	//Allocate memory used to store image modes and their corresponding regions...
	modes = new float [L*(N+2)];
	labels = new int [L];
	modePointCounts = new int [L];
	indexTable = new int [L];
	//Label image regions, also if segmentation is not to be
	//performed use the resulting classification structure to
	//calculate the image boundaries...
	Connect();
	FuseRegions(sigmaR, minRegion);
	
	for(unsigned int i = 0; i < height * width; i++) {
		unsigned char * pix = (unsigned char *)&h_segm[i];
		LUVtoRGB((float*)&h_dst[i], pix);
	}

	
	
}

void boundaries()
{
	//define the boundaries
	RegionList *regionList        = GetBoundaries();
	int        *regionIndeces     = regionList->GetRegionIndeces(0);
	int        numRegions         = regionList->GetNumRegions();
	
	numBoundaries_ = 0;
	
	
	for(int i = 0; i < numRegions; i++) {
		numBoundaries_ += regionList->GetRegionCount(i);
	}
	
	boundaries_ = new int [numBoundaries_];
	for(int i = 0; i < numBoundaries_; i++) {
		boundaries_[i] = regionIndeces[i];
	}
	
	memset(h_bndy, 0, height * width * sizeof(unsigned char));

	for(int i = 0; i < numBoundaries_; i++) {
		h_bndy[boundaries_[i]] = 255;
	}	
}




