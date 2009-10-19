/*
 *  rgbluv.cpp
 *  MeanShift
 *
 *  Created by Zvonko Krnjajic on 23.07.09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "rgbluv.h"
#include <math.h>

//data space conversion...
const cl_float Xn			= 0.95050f;
const cl_float Yn			= 1.00000f;
const cl_float Zn			= 1.08870f;
//const double Un_prime	= 0.19780;
//const double Vn_prime	= 0.46830;
const double Un_prime	= 0.19784977571475;
const double Vn_prime	= 0.46834507665248;
const double Lt			= 0.008856;

//RGB to LUV conversion
const cl_float XYZ[3][3] = {	{  0.4125f,  0.3576f,  0.1804f },
{  0.2125f,  0.7154f,  0.0721f },
{  0.0193f,  0.1192f,  0.9502f }	};

//LUV to RGB conversion
const cl_float RGB[3][3] = {	{  3.2405f, -1.5371f, -0.4985f },
{ -0.9693f,  1.8760f,  0.0416f },
{  0.0556f, -0.2040f,  1.0573f }	};



void RGBtoLUV(cl_uchar *rgb, cl_float *luv)
{
	//delcare variables
	cl_float x, y, z, L0, u_prime, v_prime, constant;
	
	//convert RGB to XYZ...
	x = XYZ[0][0]*rgb[0] + XYZ[0][1]*rgb[1] + XYZ[0][2]*rgb[2];
	y = XYZ[1][0]*rgb[0] + XYZ[1][1]*rgb[1] + XYZ[1][2]*rgb[2];
	z = XYZ[2][0]*rgb[0] + XYZ[2][1]*rgb[1] + XYZ[2][2]*rgb[2];
	//convert XYZ to LUV...
	
	//compute L*
	L0 = y / (255.0 * Yn);
	if(L0 > Lt)
		luv[0]	= (cl_float)(116.0 * (pow(L0, 1.0f/3.0f)) - 16.0);
	else
		luv[0]	= (cl_float)(903.3 * L0);
	
	//compute u_prime and v_prime
	constant	= x + 15 * y + 3 * z;
	if(constant != 0)
	{
		u_prime	= (4 * x) / constant;
		v_prime = (9 * y) / constant;
	}
	else
	{
		u_prime	= 4.0;
		v_prime	= 9.0/15.0;
	}
	
	//compute u* and v*
	luv[1] = (cl_float) (13 * luv[0] * (u_prime - Un_prime));
	luv[2] = (cl_float) (13 * luv[0] * (v_prime - Vn_prime));
	
	//done.
	return;
	
}
//define inline rounding function...
inline cl_int my_round(cl_float in_x)
{
	if (in_x < 0)
		return (cl_int)(in_x - 0.5);
	else
		return (cl_int)(in_x + 0.5);
}

void LUVtoRGB(cl_float *luv, cl_uchar *rgb)
{
	//declare variables...
	cl_int	  r, g, b;
	cl_float x, y, z, u_prime, v_prime;
	
	//perform conversion
	if(luv[0] < 0.1)
		r = g = b = 0;
	else
	{
		//convert luv to xyz...
		if(luv[0] < 8.0)
			y	= Yn * luv[0] / 903.3f;
		else
		{
			y	= (luv[0] + 16.0) / 116.0f;
			y  *= Yn * y * y;
		}
		
		u_prime	= luv[1] / (13 * luv[0]) + Un_prime;
		v_prime	= luv[2] / (13 * luv[0]) + Vn_prime;
		
		x = 9 * u_prime * y / (4 * v_prime);
		z = (12 - 3 * u_prime - 20 * v_prime) * y / (4 * v_prime);
		
		//convert xyz to rgb...
		//[r, g, b] = RGB*[x, y, z]*255.0
		r = my_round((RGB[0][0]*x + RGB[0][1]*y + RGB[0][2]*z)*255.0f);
		g = my_round((RGB[1][0]*x + RGB[1][1]*y + RGB[1][2]*z)*255.0f);
		b = my_round((RGB[2][0]*x + RGB[2][1]*y + RGB[2][2]*z)*255.0f);
		
		//check bounds...
		if(r < 0)	r = 0; if(r > 255)	r = 255;
		if(g < 0)	g = 0; if(g > 255)	g = 255;
		if(b < 0)	b = 0; if(b > 255)	b = 255;
		
	}
	
	//assign rgb values to rgb vector rgb
	rgb[0] = r;
	rgb[1] = g;
	rgb[2] = b;
	
	//done.
	return;
	
}


