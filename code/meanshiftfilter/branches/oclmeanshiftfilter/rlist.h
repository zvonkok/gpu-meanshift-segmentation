#include	<stdio.h>
#include	<assert.h>
#include	<stdlib.h>

#include <CL/cl.h>

#ifndef RLIST_H
#define RLIST_H

enum ErrorType		{NONFATAL, FATAL};

//define Region Adjacency List class prototype
class RAList 
{
public:
	
	cl_int	label;
	cl_float	edgeStrength;
	cl_int	edgePixelCount;
	RAList	*next;
	
	cl_int Insert(RAList *entry)
	{
		if(!next)  {
			next		= entry;
			entry->next = NULL;
			return 0;
		}

		if(next->label > entry->label) {
			entry->next	= next;
			next = entry;
			return 0;
		}

		//check the rest of the list...
		exists	= 0;
		cur	= next;
		while(cur) {
			if(entry->label == cur->label) {
				exists = 1;
				break;
			} else if((!(cur->next))||(cur->next->label > entry->label)) {
				entry->next	= cur->next;
				cur->next	= entry;
				break;
			}
			cur = cur->next;
		}
		return (cl_int)(exists);
	}

	
private:
	RAList	*cur, *prev;
	cl_uchar exists;
};


//define region structure
struct REGION {
	cl_int label;
	cl_int pointCount;
	cl_int region;
};

//region class prototype...
class RegionList {

public:

	RegionList(cl_int, cl_int, cl_int);
	~RegionList( void );
	void AddRegion(cl_int, cl_int, cl_int*);
	void Reset( void );	
	cl_int	GetNumRegions ( void );
	cl_int	GetLabel(cl_int);
	cl_int GetRegionCount(cl_int);
	cl_int*GetRegionIndeces(cl_int);

private:

	void ErrorHandler(char*, char*, ErrorType);
	REGION	*regionList;			//array of maxRegions regions
	cl_int minRegion;
	cl_int maxRegions;				//defines the number maximum number of regions
						//allowed (determined by user during class construction)
	cl_int numRegions;				//the number of regions currently stored by the
						//region list
	cl_int freeRegion;				//an index into the regionList pointing to the next
						//available region in the regionList

	cl_int *indexTable;			//an array of indexes that point into an external structure
						//specifying which points belong to a region
	cl_int freeBlockLoc;			//points to the next free block of memory in the indexTable
	//Dimension of data set
	cl_int N;					//dimension of data set being classified by region list
						//class
	//Length of the data set
	cl_int L;					//number of points contained by the data set being classified by
						//region list class

};

#endif



