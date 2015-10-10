/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "reference_calc.h"
#include "utils.h"
#include <iostream>

__global__ void max_reduce(const float* const d_array, float* d_max,
	const size_t elements)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;
	shared[tid] = -9999;  // 1

	if (gid < elements)
		shared[tid] = d_array[gid];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s && gid < elements)
			shared[tid] = max(shared[tid], shared[tid + s]);  // 2
		__syncthreads();
	}
	// what to do now?
	// option 1: save block result and launch another kernel
	if (tid == 0)
		d_max[blockIdx.x] = shared[tid]; // 3

}

__global__ void min_reduce(const float* const d_array, float* d_min,
	const size_t elements)
{
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;
	shared[tid] = 9999;  // 1

	if (gid < elements)
		shared[tid] = d_array[gid];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s && gid < elements)
			shared[tid] = min(shared[tid], shared[tid + s]);  // 2
		__syncthreads();
	}
	// what to do now?
	// option 1: save block result and launch another kernel
	if (tid == 0)
		d_min[blockIdx.x] = shared[tid]; // 3
}

__global__ void kernel_getHist(const float* const d_lum, long size, unsigned int* histo, int numBins, float d_min, float d_range)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= size)   return;

	unsigned int bin = static_cast<unsigned int>((d_lum[tid] - d_min) / d_range * numBins);
	if (bin > numBins - 1)
		bin = numBins - 1;


	atomicAdd(&histo[bin], 1);
}
 

__global__ void kernel_prescan(unsigned int *d_odata, unsigned int *d_idata, int n)
{
	extern __shared__ unsigned int temp[];  // allocated on invocation  
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = 1;
	
	if (thid >= n || 2*thid >= n || 2* thid + 1 >= n)   return;

	temp[2 * thid] = d_idata[2 * thid]; // load input into shared memory  
	temp[2 * thid + 1] = d_idata[2 * thid + 1];

	for (unsigned int d = n >> 1; d > 0; d >>= 1)         // build sum in place up the tree  
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	
	if (thid == 0) {
		temp[n - 1] = 0;
	} 

	for (unsigned int d = 1; d < n; d *= 2) // traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	d_odata[2 * thid] = temp[2 * thid]; // write results to device memory  
	d_odata[2 * thid + 1] = temp[2 * thid + 1];

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	float *d_max;
	float *d_min;
	float *h_max, *h_min;
	
	const int blockSize = 1024;
	const int gridSize = ceil(numCols*numRows / 1024.0);

	//printf("before max_reduce: %d %d\n", gridSize, blockSize);

	h_max = (float*)malloc(sizeof(float)*gridSize);
	h_min = (float*)malloc(sizeof(float)*gridSize);
	memset(h_max, 0, sizeof(float)*gridSize);
	memset(h_min, 0, sizeof(float)*gridSize);

	checkCudaErrors(cudaMalloc(&d_max, sizeof(float)*gridSize));
	checkCudaErrors(cudaMalloc(&d_min, sizeof(float)*gridSize));
	checkCudaErrors(cudaMemset(d_max, 0, sizeof(float)*gridSize)); //make sure no memory is left laying around
	checkCudaErrors(cudaMemset(d_min, 0, sizeof(float)*gridSize)); //make sure no memory is left laying around

	max_reduce<<<gridSize, blockSize, sizeof(float)*blockSize >>>(d_logLuminance, d_max, numRows*numCols);
	checkCudaErrors(cudaMemcpy(h_max, d_max, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
	//printf("after max_reduce: %f\n", h_max[0]);

	min_reduce <<<gridSize, blockSize, sizeof(float)*blockSize >> >(d_logLuminance, d_min, numRows*numCols);
	checkCudaErrors(cudaMemcpy(h_min, d_min, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
	//printf("after min_reduce: %f\n", h_min[0]);


	float h_max_final = -9999, h_min_final = 9999;

	for (int i = 0; i < gridSize; i++){
		if (h_max[i] > h_max_final)
			h_max_final = h_max[i];
		if (h_min[i] < h_min_final)
			h_min_final = h_min[i];
	}

	float h_range = h_max_final - h_min_final;

	//std::cerr << h_range << " " << h_max_final << " " << h_min_final << std::endl;
	
	min_logLum = h_min_final;
	max_logLum = h_max_final;

	unsigned int *d_histogram;

	checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int)*numBins));
	checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int)*numBins)); //make sure no memory is left laying around

	kernel_getHist<<<gridSize, blockSize>>>(d_logLuminance, numRows*numCols, d_histogram, numBins, h_min_final, h_range);
	unsigned int * h_histogram = (unsigned int*)malloc(sizeof(unsigned int)*numBins);
	checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

	//printf("after kernel_getHist\n");
	/*for (int i = 0; i < numBins; i++)
		printf("%d ", h_histogram[i]);
	printf("\n");
	*/
	const int gridSize2 = ceil(numBins / 1024.0);

	kernel_prescan <<<gridSize2, blockSize/2, sizeof(unsigned int)*blockSize >>>(d_cdf, d_histogram, numBins);
	unsigned int * h_cdf = (unsigned int*)malloc(sizeof(unsigned int)*numBins);
	checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

	/*
	for (int i = 0; i < numBins; i++)
		printf("%d ", h_cdf[i]);
	printf("\n");
	*/
}
