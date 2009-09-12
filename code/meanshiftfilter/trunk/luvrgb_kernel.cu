

#define Yn 1.00000f
#define Un_prime	 0.19784977571475f
#define Vn_prime	 0.46834507665248f

#define r903 1.0f/903.3f
#define r116 1.0f/116.0f

//define inline rounding function...
__device__ int my_round(float in_x)
{
	if (in_x < 0)
		return (int)(in_x - 0.5);
	else
		return (int)(in_x + 0.5);
}

// convert floating point rgba color to 32-bit integer
__device__ uint rgbaFloatToInt(float4 rgba)
{
   
    return (uint(rgba.w)<<24) | (uint(rgba.z)<<16) | (uint(rgba.y)<<8) | uint(rgba.x);
}


__global__ void luvtorgb(float4 *d_luv, unsigned int *d_rgb, unsigned int width)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int i = ix + iy * width;
	
	float4 luv = d_luv[i];
	
	float RGB[3][3] = {
		{  3.2405f, -1.5371f, -0.4985f },
		{ -0.9693f,  1.8760f,  0.0416f },
		{  0.0556f, -0.2040f,  1.0573f }	
	};

	//declare variables...
	float r, g, b;
	float x, y, z;
	float u_prime, v_prime;
	
	
	//convert luv to xyz...
	if(luv.x < 8.0)
		y = Yn * luv.x * r903;
	else
	{
		y = (luv.x + 16.0f) * r116;
		y *= Yn * y * y;
	}
	
	u_prime	= luv.y / (13.0f * luv.x) + Un_prime;
	v_prime	= luv.z / (13.0f * luv.x) + Vn_prime;
	
	x = 9.0f * u_prime * y / (4.0f * v_prime);
	z = (12.0f - 3.0f * u_prime - 20.0f * v_prime) * y / (4.0f * v_prime);
	
	//convert xyz to rgb...
	//[r, g, b] = RGB*[x, y, z]*255.0
	
	float r_0 = RGB[0][0]*x;
	float g_0 = RGB[1][0]*x;
	float b_0 = RGB[2][0]*x;
	
	float r_1 = RGB[0][1]*y;
	float g_1 = RGB[1][1]*y;
	float b_1 = RGB[2][1]*y;
	
	float r_2 = RGB[0][2]*z;
	float g_2 = RGB[1][2]*z;
	float b_2 = RGB[2][2]*z;
	
	r = my_round((r_0 + r_1 + r_2) * 255.0f);
	g = my_round((g_0 + g_1 + g_2) * 255.0f);
	b = my_round((b_0 + b_1 + b_2) * 255.0f);
	
	if(r < 0)	r = 0; if(r > 255)	r = 255;
	if(g < 0)	g = 0; if(g > 255)	g = 255;
	if(b < 0)	b = 0; if(b > 255)	b = 255;
	
	
	
	float4 rgba = { r, g, b, 0.0f };
	d_rgb[i] = rgbaFloatToInt(rgba);
	return;

}

extern "C" void luvToRgb(dim3 grid, dim3 threads, float4* d_luv, 
	unsigned int* d_rgb, unsigned int width)
{
	luvtorgb<<< grid, threads >>>(d_luv, d_rgb, width);
}


