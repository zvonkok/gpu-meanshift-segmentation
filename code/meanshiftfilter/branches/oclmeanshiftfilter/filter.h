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

#include <CL/cl.h>

extern "C" void Filter();

extern cl_uint * h_filt;
extern cl_float4 * h_src;
extern cl_float4 * h_dst;

extern cl_uint height;
extern cl_uint width;

extern cl_uint N;
extern cl_uint L;

extern cl_float sigmaS;
extern cl_float sigmaR;

#endif // FILTER_H

