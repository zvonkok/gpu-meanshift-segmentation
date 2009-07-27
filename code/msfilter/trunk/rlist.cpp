#include	"rlist.h"
#include	<stdio.h>
#include	<stdlib.h>


RegionList::RegionList(int maxRegions_, int L_, int N_)
{
	if((maxRegions = maxRegions_) <= 0)
		ErrorHandler("RegionList", "Maximum number of regions is zero or negative.", FATAL);
	if((N = N_) <= 0)
		ErrorHandler("RegionList", "Dimension is zero or negative.", FATAL);
	if((L = L_) <= 0)
		ErrorHandler("RegionList", "Length of data set is zero or negative.", FATAL);
	if(!(indexTable = new int [L]))
		ErrorHandler("RegionList", "Not enough memory.", FATAL);
	if(!(regionList = new REGION [maxRegions]))
		ErrorHandler("RegionList", "Not enough memory.", FATAL);
	numRegions		= freeRegion = 0;
	freeBlockLoc	= 0;
}

RegionList::~RegionList( void )
{
	delete [] regionList;
	delete [] indexTable;
}

void RegionList::AddRegion(int label, int pointCount, int *indeces)
{

	//make sure that there is enough room for this new region 
	//in the region list array...
	if(numRegions >= maxRegions)
		ErrorHandler("AddRegion", "Not enough memory allocated.", FATAL);

	//make sure that label is positive and point Count > 0...
	if((label < 0)||(pointCount <= 0))
		ErrorHandler("AddRegion", "Label is negative or number of points in region is invalid.", FATAL);

	//make sure that there is enough memory in the indexTable
	//for this region...
	if((freeBlockLoc + pointCount) > L)
		ErrorHandler("AddRegion", "Adding more points than what is contained in data set.", FATAL);

	//place new region into region list array using
	//freeRegion index
	regionList[freeRegion].label		= label;
	regionList[freeRegion].pointCount	= pointCount;
	regionList[freeRegion].region		= freeBlockLoc;

	//copy indeces into indexTable using freeBlock...
	int i;
	for(i = 0; i < pointCount; i++)
		indexTable[freeBlockLoc+i] = indeces[i];

	//increment freeBlock to point to the next free
	//block
	freeBlockLoc	+= pointCount;

	//increment freeRegion to point to the next free region
	//also, increment numRegions to indicate that another
	//region has been added to the region list
	freeRegion++;
	numRegions++;
}


void RegionList::Reset( void )
{
	freeRegion = numRegions = freeBlockLoc = 0;
}

int RegionList::GetNumRegions( void )
{
	return numRegions;
}

int RegionList::GetLabel(int regionNum)
{
	return regionList[regionNum].label;
}


int RegionList::GetRegionCount(int regionNum)
{
	return regionList[regionNum].pointCount;
}


int *RegionList::GetRegionIndeces(int regionNum)
{
	return &indexTable[regionList[regionNum].region];
}

void RegionList::ErrorHandler(char *functName, char* errmsg, ErrorType status)
{
	if(status == NONFATAL)
		fprintf(stderr, "\n%s Error: %s\n", functName, errmsg);
	else
	{
		fprintf(stderr, "\n%s Fatal Error: %s\n\nAborting Program.\n\n", functName, errmsg);
		exit(1);
	}

}


