#include	<stdio.h>
#include	<assert.h>
#include	<stdlib.h>

#ifndef RLIST_H
#define RLIST_H

enum ErrorType		{NONFATAL, FATAL};

//define Region Adjacency List class prototype
class RAList 
{
public:
	
	int	label;
	float	edgeStrength;
	int	edgePixelCount;
	RAList	*next;
	
	int Insert(RAList *entry)
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
		return (int)(exists);
	}

	
private:
	RAList	*cur, *prev;
	unsigned char exists;
};


//define region structure
struct REGION {
	int label;
	int pointCount;
	int region;
};

//region class prototype...
class RegionList {

public:

	RegionList(int, int, int);
	~RegionList( void );
	void AddRegion(int, int, int*);
	void Reset( void );	
	int	GetNumRegions ( void );
	int	GetLabel(int);
	int GetRegionCount(int);
	int*GetRegionIndeces(int);

private:

	void ErrorHandler(char*, char*, ErrorType);
	REGION	*regionList;			//array of maxRegions regions
	int minRegion;
	int maxRegions;				//defines the number maximum number of regions
						//allowed (determined by user during class construction)
	int numRegions;				//the number of regions currently stored by the
						//region list
	int freeRegion;				//an index into the regionList pointing to the next
						//available region in the regionList

	int *indexTable;			//an array of indexes that point into an external structure
						//specifying which points belong to a region
	int freeBlockLoc;			//points to the next free block of memory in the indexTable
	//Dimension of data set
	int N;					//dimension of data set being classified by region list
						//class
	//Length of the data set
	int L;					//number of points contained by the data set being classified by
						//region list class

};

#endif



