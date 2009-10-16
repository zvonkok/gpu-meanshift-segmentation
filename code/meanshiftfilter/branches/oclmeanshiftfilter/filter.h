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

#include <cuda_runtime_api.h>

extern "C" void Filter();

extern unsigned int * h_filt;
extern float4 * h_src;
extern float4 * h_dst;

extern unsigned int height;
extern unsigned int width;

extern unsigned int N;
extern unsigned int L;

extern float sigmaS;
extern float sigmaR;

#endif // FILTER_H

