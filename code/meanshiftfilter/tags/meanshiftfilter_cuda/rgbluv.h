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

extern "C" void RGBtoLUV(unsigned char *rgb, float *luv);
extern "C" inline int my_round(float in_x);
extern "C" void LUVtoRGB(float *luv, unsigned char *rgb);

#endif // RGBLUV_H