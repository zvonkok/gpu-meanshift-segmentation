/*
 *  rgbluv.h
 *  MeanShift
 *
 *  Created by Zvonko Krnjajic on 23.07.09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef RGBLUV_H
#define RGBLUV_H

#include <CL/cl.h>

extern "C" void RGBtoLUV(cl_uchar *rgb, cl_float *luv);
extern "C" inline cl_int my_round(cl_float in_x);
extern "C" void LUVtoRGB(cl_float *luv, cl_uchar *rgb);

#endif // RGBLUV_H