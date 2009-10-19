

#define Yn 1.00000f
#define Un_prime	0.19784977571475f
#define Vn_prime 0.46834507665248f
#define Lt 0.008856f


__kernel cl_float4 rgbaIntToFloat(uint c)
{
    cl_float4 rgba;
    rgba[0] = (c & 0xff);
    rgba[1] = ((c>>8) & 0xff) ;
    rgba[2] = ((c>>16) & 0xff);
    rgba[3] = ((c>>24) & 0xff);
    return rgba;
}
__global__ void rgbtoluv(cl_float4 *d_luv, unsigned int *d_img, unsigned int width)
{
	float XYZ[3][3] = {
		{  0.4125f,  0.3576f,  0.1804f },
		{  0.2125f,  0.7154f,  0.0721f },
		{  0.0193f,  0.1192f,  0.9502f }	
	};
	
	int ix = blockIdx[0] * blockDim[0] + threadIdx[0];
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int i = ix + iy * width;
	
	unsigned int rgb = d_img[i];
	
	cl_float4 rgba = rgbaIntToFloat(rgb);
	cl_float4 luv;
	
	
	//convert RGB to XYZ...
	float x = XYZ[0][0]*rgba[0] + XYZ[0][1]*rgba[1] + XYZ[0][2]*rgba[2];
	float y = XYZ[1][0]*rgba[0] + XYZ[1][1]*rgba[1] + XYZ[1][2]*rgba[2];
	float z = XYZ[2][0]*rgba[0] + XYZ[2][1]*rgba[1] + XYZ[2][2]*rgba[2];
	//convert XYZ to LUV...
	
	//compute L*
	float L0 = y / (255.0 * Yn);
	if(L0 > Lt)
		luv[0]	= (float)(116.0 * (pow(L0, 1.0f/3.0f)) - 16.0);
	else
		luv[0]	= (float)(903.3 * L0);
	
	//compute u_prime and v_prime
	float u_prime;
	float v_prime;
	float constant	= x + 15 * y + 3 * z;
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
	luv[1] = (float) (13 * luv[0] * (u_prime - Un_prime));
	luv[2] = (float) (13 * luv[0] * (v_prime - Vn_prime));
	
	
	d_luv[i] = luv;
}

extern "C" void rgbToLuv(dim3 grid, dim3 threads, cl_float4* d_luv, 
	unsigned int* d_img, unsigned int width)
{
	rgbtoluv<<< grid, threads >>>(d_luv, d_img, width);
}


