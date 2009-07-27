/*
 *  filter.h
 *  MeanShift
 *
 *  Created by Zvonko Krnjajic on 23.07.09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FILTER_H
#define FILTER_H

const float	EPSILON		= 0.01;	// define threshold (approx. Value of Mh at a peak or plateau)
const int		LIMIT       = 100;	// define max. # of iterations to find mode

extern "C" void Filter();

extern unsigned int * h_filt;
extern float * h_src;
extern float * h_dst;

extern unsigned int height;
extern unsigned int width;

extern unsigned int N;
extern unsigned int L;

extern float sigmaS;
extern float sigmaR;

#endif // FILTER_H

