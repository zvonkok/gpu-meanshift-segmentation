

#define Yn 1.00000f
#define Un_prime	0.19784977571475f
#define Vn_prime 0.46834507665248f
#define Lt 0.008856f


__device__ float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff);
    rgba.y = ((c>>8) & 0xff) ;
    rgba.z = ((c>>16) & 0xff);
    rgba.w = ((c>>24) & 0xff);
    return rgba;
}


__global__ void rgbtoluv(float4 *d_luv, unsigned int *d_img, unsigned int width)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int i = ix + iy * width;
	
	unsigned int rgb = d_img[i];
	float4 rgba = rgbaIntToFloat(rgb);
	
	float4 luv;
	
	float XYZ[3][3] = {
		{  0.4125f,  0.3576f,  0.1804f },
		{  0.2125f,  0.7154f,  0.0721f },
		{  0.0193f,  0.1192f,  0.9502f }	
	};
	
	
	//delcare variables
	float L0, u_prime, v_prime;
	
	//convert RGB to XYZ...
	float x_0 = XYZ[0][0] * rgba.x;
	float y_0 = XYZ[1][0] * rgba.x; 
	float z_0 = XYZ[2][0] * rgba.x; 
	
	float x_1 = XYZ[0][1] * rgba.y; 
	float y_1 = XYZ[1][1] * rgba.y; 
	float z_1 = XYZ[2][1] * rgba.y; 
	
	float x_2 = XYZ[0][2] * rgba.z;
	float y_2 = XYZ[1][2] * rgba.z;
	float z_2 = XYZ[2][2] * rgba.z;
	
	float x = x_0 + x_1 + x_2;
	float y = y_0 + y_1 + y_2;
	float z = z_0 + z_1 + z_2;
	//convert XYZ to LUV...
	
	//compute L*
	L0 = y / (255.0f * Yn);
	
	if(L0 > Lt)
		luv.x	= 116.0f * (__powf(L0, 0.3333333333333333333f)) - 16.0f;
	else
		luv.x	= 903.3f * L0;
	
	
	//compute u_prime and v_prime
	//float constant	= x + 15 * y + 3 * z;
	float c_0 = 15.0f * y;
	float c_1 = 3.0f * z;
	float c = x + c_0 + c_1;
	
	if(c != 0)
	{
		u_prime	= (4.0f * x) / c;
		v_prime = (9.0f * y) / c;
	}
	else
	{
		u_prime	= 4.0f;
		v_prime	= 0.6f;
	}
	
	
	//compute u* and v*
	luv.y = 13.0f * luv.x * (u_prime - Un_prime);
	luv.z = 13.0f * luv.x * (v_prime - Vn_prime);
	
	d_luv[i] = luv;
	
	return;
	
}

extern "C" void rgbToLuv(dim3 grid, dim3 threads, float4* d_luv, 
	unsigned int* d_img, unsigned int width)
{
	rgbtoluv<<< grid, threads >>>(d_luv, d_img, width);
}


