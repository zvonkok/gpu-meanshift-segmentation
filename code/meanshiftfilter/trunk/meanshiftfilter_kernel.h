#define INNER_LOOP			\
diff0 = 0.0f;				\
					\
dx = (k - yk[0]) * rsigmaS;		\
dy = (j - yk[1]) * rsigmaS;		\
					\
diff0 += dx * dx;			\
diff0 += dy * dy;			\
					\
					\
if (diff0 >= 1.0f) continue;		\
					\
diff1 = 0.0f;				\
					\
luv = tex2D(tex, k, j); 			\
					\
dl = (luv.x - yk[2]) * rsigmaR;           	\
du = (luv.y - yk[3]) * rsigmaR;           	\
dv = (luv.z - yk[4]) * rsigmaR;           	\
					\
diff1 += dl * dl;			\
					\
if((yk[2] > 80.0f)) 			\	
	diff1 += 3.0f * dl * dl;		\
					\
diff1 += du * du;			\
diff1 += dv * dv;			\
					\
if (diff1 >= 1.0f) continue;		\
					\
Mh[0] += k;				\
Mh[1] += j;				\
Mh[2] += luv.x;				\
Mh[3] += luv.y;				\
Mh[4] += luv.z;				\
wsum += 1.0f; //weight

/* ORIGINAL

diff0 = 0.0f;

//Determine if inside search window
//Calculate distance squared of sub-space s	

dx = (k - yk[0]) * rsigmaS;
dy = (j - yk[1]) * rsigmaS;

diff0 += dx * dx;
diff0 += dy * dy;


if (diff0 >= 1.0f) continue;		


diff1 = 0.0f;
//get index into data array
luv = tex2D(tex, k, j); //luv = d_src[i * width + j];

dl = (luv.x - yk[2]) * rsigmaR;               
du = (luv.y - yk[3]) * rsigmaR;               
dv = (luv.z - yk[4]) * rsigmaR;               


diff1 += dl * dl;

if((yk[2] > 80.0f)) 
diff1 += 3.0f * dl * dl;

diff1 += du * du;
diff1 += dv * dv;


if (diff1 >= 1.0f) continue;


// If its inside search window perform sum and count
// For a uniform kernel weight == 1 for all feature points
// considered point is within sphere => accumulate to mean
Mh[0] += k;
Mh[1] += j;
Mh[2] += luv.x;
Mh[3] += luv.y;
Mh[4] += luv.z;
wsum += 1.0f; //weight
*/

