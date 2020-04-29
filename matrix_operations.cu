//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 1
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

  
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <algorithm>

using namespace std;

class Matrix{
public:
    int m=0;							////number of rows
    int n=0;							////number of columns
	vector<float> elements_on_host;		////we use a std::vector for the element array on host
    float* elements_on_dev=0;			////we use a pointer for the element array on device
	bool on_host=true;

	////constructors
	__host__ Matrix(){}

	__host__ Matrix(const int _m,const int _n,bool _on_host=true)
	{
		on_host=_on_host;
		if(on_host)Resize_On_Host(_m,_n);
		else Resize_On_Device(_m,_n);
	}

	////destructor
	__host__ ~Matrix()
	{
		if(!on_host&&elements_on_dev!=0) cudaFree(elements_on_dev);		
	}

	////Resize on host or device
	__host__ void Resize_On_Host(const int _m,const int _n)
	{
		if(m==_m&&n==_n)return;
		m=_m;
		n=_n;
		elements_on_host.resize(m*n);
	}

	__host__ void Resize_On_Device(const int _m,const int _n)
	{
		if(m==_m&&n==_n)return;
		m=_m;
		n=_n;
		if(elements_on_dev!=0)cudaFree(elements_on_dev);
		cudaMalloc((void**)&elements_on_dev,m*n*sizeof(float));
	}

	////random access a matrix element
	inline __host__ float& operator() (const int i,const int j)
	{
		return elements_on_host[i*n+j];
	}

	inline __host__ const float& operator() (const int i,const int j) const
	{
		return elements_on_host[i*n+j];
	}

	////copy data with four cases (CPU->CPU, GPU->CPU, GPU->GPU, CPU->GPU)
	__host__ Matrix& operator= (const Matrix& mtx)
	{
		if(on_host&&mtx.on_host){
			Resize_On_Host(mtx.m,mtx.n);
			elements_on_host=mtx.elements_on_host;
		}
		else if(on_host&&!mtx.on_host){
			Resize_On_Host(mtx.m,mtx.n);
			cudaMemcpy(&elements_on_host[0],mtx.elements_on_dev,m*n*sizeof(float),cudaMemcpyDeviceToHost);
		}
		else if(!on_host&&!mtx.on_host){
			Resize_On_Device(mtx.m,mtx.n);
			cudaMemcpy(elements_on_dev,mtx.elements_on_dev,mtx.m*n*sizeof(float),cudaMemcpyDeviceToDevice);
		}
		else if(!on_host&&mtx.on_host){
			Resize_On_Device(mtx.m,mtx.n);
			cudaMemcpy(elements_on_dev,&mtx.elements_on_host[0],m*n*sizeof(float),cudaMemcpyHostToDevice);
		}
		return *this;
	}

	////print matrix elements on screen
	__host__ friend ostream & operator << (ostream &out,const Matrix &mtx)
	{
		if(!mtx.on_host)
			cout<<"Print for matrix on device is not supported."<<endl;

		for(int i=0;i<mtx.m;i++){
			for(int j=0;j<mtx.n;j++){
				out<<mtx(i,j)<<", ";
			}
			out<<std::endl;
		}
		return out;
	}
};

__global__ void Transpose2(const float* Ae,float* ATe,const int Am,const int An)
{
	__shared__ int s[16][17];

	int i=blockIdx.x*blockDim.x+threadIdx.y;
	int j=blockIdx.y*blockDim.y+threadIdx.x;

	// TODO: Am or An; Should be An(width) not Am?... Also modified some index
	s[threadIdx.x][threadIdx.y]=Ae[i*An+j];  
	__syncthreads();

	int ti=blockIdx.y*blockDim.x+threadIdx.y;
	int tj=blockIdx.x*blockDim.y+threadIdx.x;
	ATe[ti*Am+tj]=s[threadIdx.y][threadIdx.x];
} 


template <int BLOCK_SIZE, bool MULTIPLE_OF_BLOCKSIZE> 
__global__ void Matrix_Multiplication_Kernel_Your_Version(const float* Ae,const float* Be,float* Ce,const int Am,const int An,const int Bn)
{
	int a_start = An * BLOCK_SIZE*blockIdx.x;
	int a_end = a_start + An - 1;
	int b_start = BLOCK_SIZE*blockIdx.y;
	int b_BLOCK_SIZE = BLOCK_SIZE * Bn;

	__shared__ float A_sub[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float B_sub[BLOCK_SIZE][BLOCK_SIZE];

	int a_t = An * threadIdx.y + threadIdx.x;
	int b_t = Bn * threadIdx.y + threadIdx.x;

	float val = 0.0f;

	for (int a = a_start, b = b_start; a <= a_end; a += BLOCK_SIZE, b += b_BLOCK_SIZE) {
		if (MULTIPLE_OF_BLOCKSIZE || a + a_t < Am * An) { 
			A_sub[threadIdx.y][threadIdx.x] = Ae[a + a_t]; 
		} else { 
			A_sub[threadIdx.y][threadIdx.x] = 0.0f; 
		}
		if (MULTIPLE_OF_BLOCKSIZE || b + b_t < An * Bn ) { 
			B_sub[threadIdx.y][threadIdx.x] = Be[b + b_t]; 
		} else { 
			B_sub[threadIdx.y][threadIdx.x] = 0.0f; 
		}
		__syncthreads();

	#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			val += A_sub[threadIdx.y][k] * B_sub[k][threadIdx.x];
		}
		__syncthreads();
	}

	int i=blockIdx.x*blockDim.x+threadIdx.y;
	int j=blockIdx.y*blockDim.y+threadIdx.x;
	if (MULTIPLE_OF_BLOCKSIZE || i < Am && j < Bn) {
		Ce[i*Bn+j]=val;
	}
}

template <int BLOCK_SIZE, bool MULTIPLE_OF_BLOCKSIZE> 
__global__ void Matrix_Multiplication_ATBA_Kernel_Your_Version(const float* ATe, const float* Ae,const float* Be,float* Ce,const int Am,const int An)
{
	__shared__ float AT_sub[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float B_sub[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float A_sub[BLOCK_SIZE][BLOCK_SIZE];

	int a_start = Am * BLOCK_SIZE*blockIdx.x;
	int a_end = a_start + Am - 1;
	int b_start = BLOCK_SIZE*blockIdx.y;
	int b_BLOCK_SIZE = BLOCK_SIZE * Am;

	int a_t = Am * threadIdx.y + threadIdx.x;
	int b_t = Am * threadIdx.y + threadIdx.x;
	float val = 0.0f;
	for (int a = a_start, b = b_start; a <= a_end; a += BLOCK_SIZE, b += b_BLOCK_SIZE) {
		AT_sub[threadIdx.y][threadIdx.x] = ATe[a + a_t]; 
		B_sub[threadIdx.y][threadIdx.x] = Be[b + b_t]; 
		__syncthreads();

	    #pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			val += AT_sub[threadIdx.y][k] * B_sub[k][threadIdx.x];
		}
		__syncthreads();
	}
	AT_sub[threadIdx.y][threadIdx.x]=val;

	int start = An * BLOCK_SIZE*blockIdx.y;
	int end = start + An - 1;
	int a_tt = An * threadIdx.y + threadIdx.x;
	for (int a = start, a_col = 0; a <= end; a += BLOCK_SIZE, ++a_col) {
		A_sub[threadIdx.y][threadIdx.x] = ATe[a + a_tt]; 
		__syncthreads();
		
		float value = 0.0f;
	    #pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			value += AT_sub[threadIdx.y][k] * A_sub[k][threadIdx.x];
		}
		__syncthreads();

		int row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
		int col = a_col * BLOCK_SIZE + threadIdx.x;
		atomicAdd(&(Ce[An * row + col]), value);
	}
}

// initial C to be zeros
template<bool MULTIPLE_OF_BLOCKSIZE> 
__global__ void initial_zeros(float * C, int width, int height) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.y;
	int j = blockDim.y * blockIdx.y + threadIdx.x;
	if (MULTIPLE_OF_BLOCKSIZE ||(i < height && j < width)) {
		C[i*width + j] = 0;
	}
}


template <unsigned int blockSize, bool first_reduction>
__global__ void
reduce_kernel(float *a_in, float *a_out, unsigned int n)
{
	extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
	float sum = 0;
	// if (i == 0) { printf("gridSize %d\tn %d\n", gridSize, n); }
    while (i < n) {
		if (first_reduction) {
			float temp = a_in[i]; sum += temp * temp;
		}
		else {
			sum += a_in[i];
		}
        if (i + blockSize < n) {
			if (first_reduction) {
				float temp = a_in[i+blockSize]; sum += temp * temp;
			}
			else {
				sum += a_in[i+blockSize];
			}
		}
		i += gridSize;
    }
    sdata[tid] = sum;
    __syncthreads();

    if ((blockSize >= 512) && (tid < 256)) {
		sdata[tid] += sdata[tid + 256];
    } __syncthreads();

    if ((blockSize >= 256) &&(tid < 128)) {
		sdata[tid] += sdata[tid + 128];
    } __syncthreads();

    if ((blockSize >= 128) && (tid <  64)) {
		sdata[tid] += sdata[tid + 64];
    } __syncthreads();

    if ((blockSize >=  64) && (tid < 32)) {
        sdata[tid] += sdata[tid + 32];
    } __syncthreads();

    if ((blockSize >=  32) && (tid < 16)) {
        sdata[tid] += sdata[tid + 16];
    } __syncthreads();

    if ((blockSize >=  16) && (tid <  8)) {
        sdata[tid] += sdata[tid + 8];
    } __syncthreads();

    if ((blockSize >=   8) && (tid <  4)) {
        sdata[tid] += sdata[tid + 4];
    } __syncthreads();

    if ((blockSize >=   4) && (tid <  2)) {
        sdata[tid] += sdata[tid + 2];
    } __syncthreads();

    if ((blockSize >=   2) && ( tid <  1)) {
		sdata[tid] += sdata[tid + 1];
    } __syncthreads();

    if (tid == 0) a_out[blockIdx.x] = sdata[0]; 
}

template <bool first_reduction>
void reduce(int size, int threads, int blocks, float *a_in, float *a_out)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = threads * sizeof(float);

	switch (threads)
	{
		case 512:
			reduce_kernel<512, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;

		case 256:
			reduce_kernel<256, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;

		case 128:
			reduce_kernel<128, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;

		case 64:
			reduce_kernel<64, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;

		case 32:
			reduce_kernel<32, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;

		case 16:
			reduce_kernel<16, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;

		case  8:
			reduce_kernel<8, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;

		case  4:
			reduce_kernel<4, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;

		case  2:
			reduce_kernel<2, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;

		case  1:
			reduce_kernel<1, first_reduction><<< dimGrid, dimBlock, smemSize >>>(a_in, a_out, size);
			break;
	}
}

void setBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
	threads = maxThreads;
	if (n < maxThreads*2) {  
		int temp = (n + 1)/ 2;
		unsigned int n = 1<<10;
		for (; n>=32; n >>= 1) {  
			if (temp > n) {break;}
		}
		threads = n << 1; 
	}
	blocks = min(maxBlocks, (n + (threads * 2 - 1)) / (threads * 2));
	// cout << blocks << " " << threads << endl;
}

float runReduce(int size, int maxThreads, int maxBlocks, float *a_in)
{
	int numBlocks = 0, numThreads = 0;
	setBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);
	float * a_out = nullptr; cudaMalloc((void **) &a_out, numBlocks*sizeof(float));
	reduce<true>(size, numThreads, numBlocks, a_in, a_out);
	size = numBlocks;

	float result = 0;
	int cpuNum = 16;
	while (size > cpuNum) {
		swap(a_in, a_out);
		// cudaMemcpy(a_in, a_out, size*sizeof(float), cudaMemcpyDeviceToDevice);
		setBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);
		reduce<false>(size, numThreads, numBlocks, a_in, a_out);
		size = numBlocks;
	} 
	if (size > 1) {  
		float * h_odata = (float *)malloc (size * sizeof(float));
		cudaMemcpy(h_odata, a_out, size * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i=0; i < size; i++) { result += h_odata[i]; }
		free(h_odata);
	} else {  
		cudaMemcpy(&result, a_out, sizeof(float), cudaMemcpyDeviceToHost);
	}

	cudaFree(a_out);
    return result;
}


__host__ void Test_Transpose_On_GPU(const Matrix& A,Matrix& AT)
{
	//// Load A to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;

	//// Allocate AT in device memory
	Matrix AT_on_dev(A_on_dev.n,A_on_dev.m,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	const int block_size=16;
	const int block_num_x=A.m/block_size;
	const int block_num_y=A.n/block_size;

	////TODO: this is a sample implementation. Comment it out to test your own code.
	Transpose2<<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
		(A_on_dev.elements_on_dev,AT_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for transpose: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//// Transfer data back to CPU
	AT=AT_on_dev;

	// bool if_break = true;
	// bool flag = false;
	// for (int i = 0; i < A.m; i++) {
	// 	for (int j = 0; j < A.n; j++) {
	// 		if (A(i, j) != AT(j, i)) {
	// 			cout << "[WRONG RESULT]  transpose"  << "  " <<  i << " " << j << " " << A(i, j) << endl;
	// 			flag = true; 
	// 		} if (if_break && flag) break;
	// 	} if (if_break && flag) break;
	// }
}

__host__ void Test_Matrix_Multiplication_AB_On_GPU(const Matrix& A,const Matrix& B,Matrix& C)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	Matrix B_on_dev(B.m,B.n,false);
	B_on_dev=B;

	//// Allocate C in device memory
	Matrix C_on_dev(A_on_dev.m,B_on_dev.n,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	const int block_size= 32;
	const int block_num_x=(C.m-1)/block_size+1;
	const int block_num_y=(C.n-1)/block_size+1;
	const bool multiple_of_blocksize = ((A.m & (block_size - 1)) | (A.n & (block_size - 1)) | (B.n & (block_size - 1))) == 0;
	// cout << multiple_of_blocksize << endl << block_size << endl << block_num_x << endl << block_num_y << endl;

	if (multiple_of_blocksize) {
	Matrix_Multiplication_Kernel_Your_Version<block_size, true><<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
		(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n,B_on_dev.n);
	} else {
	Matrix_Multiplication_Kernel_Your_Version<block_size, false><<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
		(A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n,B_on_dev.n);
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for matrix multiplication AB: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//// Transfer data back to CPU
	C=C_on_dev;
}

__host__ void Test_Matrix_Multiplication_ATBA_On_GPU_TwiceMultiplication(const Matrix& A,const Matrix& B,Matrix& C)
{
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	Matrix B_on_dev(B.m,B.n,false);
	B_on_dev=B;

	//// Allocate C in device memory
	Matrix C_on_dev(A_on_dev.n,A_on_dev.n,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	// Matrix h_AT(A.n, A.m);
	// Test_Transpose_On_GPU(A,h_AT);
	// Matrix h_ATB(A.n, A.m);
	// Test_Matrix_Multiplication_AB_On_GPU(h_AT,B,h_ATB);
	// Test_Matrix_Multiplication_AB_On_GPU(h_ATB,A,C);

	Matrix AT_on_dev(A_on_dev.n,A_on_dev.m,false);
	const int block_size_t=16;
	const int block_num_x_t=A.m/block_size_t;
	const int block_num_y_t=A.n/block_size_t;
	Transpose2<<<dim3(block_num_x_t,block_num_y_t),dim3(block_size_t,block_size_t)>>>
		(A_on_dev.elements_on_dev,AT_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);

	// TODO: I should write a warpper function for this kernel call...
	Matrix dev_ATB(A.n, A.m, false);
	const int block_size1= 32;
	const int block_num_x1=(dev_ATB.m-1)/block_size1+1;
	const int block_num_y1=(dev_ATB.n-1)/block_size1+1;
	const bool multiple_of_blocksize1 = ((AT_on_dev.m & (block_size1 - 1)) | (AT_on_dev.n & (block_size1 - 1)) | (B_on_dev.n & (block_size1 - 1))) == 0;
	if (multiple_of_blocksize1) {
	Matrix_Multiplication_Kernel_Your_Version<block_size1, true><<<dim3(block_num_x1,block_num_y1),dim3(block_size1,block_size1)>>>
		(AT_on_dev.elements_on_dev,B_on_dev.elements_on_dev,dev_ATB.elements_on_dev,AT_on_dev.m,AT_on_dev.n,B_on_dev.n);
	} else {
	Matrix_Multiplication_Kernel_Your_Version<block_size1, false><<<dim3(block_num_x1,block_num_y1),dim3(block_size1,block_size1)>>>
		(AT_on_dev.elements_on_dev,B_on_dev.elements_on_dev,dev_ATB.elements_on_dev,AT_on_dev.m,AT_on_dev.n,B_on_dev.n);
	}

	const int block_size2= 32;
	const int block_num_x2=(C_on_dev.m-1)/block_size2+1;
	const int block_num_y2=(C_on_dev.n-1)/block_size2+1;
	const bool multiple_of_blocksize2 = ((dev_ATB.m & (block_size2 - 1)) | (dev_ATB.n & (block_size2 - 1)) | (A_on_dev.n & (block_size2 - 1))) == 0;
	if (multiple_of_blocksize2) {
	Matrix_Multiplication_Kernel_Your_Version<block_size2, true><<<dim3(block_num_x2,block_num_y2),dim3(block_size2,block_size2)>>>
		(dev_ATB.elements_on_dev,A_on_dev.elements_on_dev,C_on_dev.elements_on_dev,dev_ATB.m,dev_ATB.n,A_on_dev.n);
	} else {
	Matrix_Multiplication_Kernel_Your_Version<block_size2, false><<<dim3(block_num_x2,block_num_y2),dim3(block_size2,block_size2)>>>
		(dev_ATB.elements_on_dev,A_on_dev.elements_on_dev,C_on_dev.elements_on_dev,dev_ATB.m,dev_ATB.n,A_on_dev.n);
	}
	
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for matrix multiplication ATBA: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	C = C_on_dev;
}

__host__ void Test_Matrix_Multiplication_ATBA_On_GPU(const Matrix& A,const Matrix& B,Matrix& C)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	Matrix B_on_dev(B.m,B.n,false);
	B_on_dev=B;
	
	//// Allocate C in device memory
	Matrix C_on_dev(A_on_dev.n,A_on_dev.n,false);

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel

	const int block_size_c= 32;
	const int block_num_x_c=(C.m-1)/block_size_c+1;
	const int block_num_y_c=(C.n-1)/block_size_c+1;
	const bool multiple_of_blocksize_1 = (C.m & (block_size_c - 1)) == 0;
	if (multiple_of_blocksize_1) {
		initial_zeros<true><<<dim3(block_num_x_c,block_num_y_c),dim3(block_size_c, block_size_c)>>>
			(C_on_dev.elements_on_dev,C_on_dev.n,C_on_dev.m);
		} else {
		initial_zeros<false><<<dim3(block_num_x_c,block_num_y_c),dim3(block_size_c, block_size_c)>>>
			(C_on_dev.elements_on_dev,C_on_dev.n,C_on_dev.m);
	}

	Matrix AT_on_dev(A_on_dev.n,A_on_dev.m,false);
	const int block_size_t=16;
	const int block_num_x_t=A.m/block_size_t;
	const int block_num_y_t=A.n/block_size_t;
	Transpose2<<<dim3(block_num_x_t,block_num_y_t),dim3(block_size_t,block_size_t)>>>
		(A_on_dev.elements_on_dev,AT_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);


	const int block_size= 32;
	const int block_num_x=(AT_on_dev.m-1)/block_size+1;
	const int block_num_y=(AT_on_dev.n-1)/block_size+1;
	const bool multiple_of_blocksize_2 = ((A.m & (block_size - 1)) | (A.n & (block_size - 1))) == 0;
	if (multiple_of_blocksize_2) {
	Matrix_Multiplication_ATBA_Kernel_Your_Version<block_size, true><<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
		(AT_on_dev.elements_on_dev, A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);
	} else {
		cout << "This function currently can not deal with dimensions that are not a multiple of block_size." << endl;
	// Matrix_Multiplication_ATBA_Kernel_Your_Version<block_size, false><<<dim3(block_num_x,block_num_y),dim3(block_size,block_size)>>>
	// 	(AT_on_dev.elements_on_dev, A_on_dev.elements_on_dev,B_on_dev.elements_on_dev,C_on_dev.elements_on_dev,A_on_dev.m,A_on_dev.n);
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for matrix multiplication ATBA: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	//// Transfer data back to CPU
	C=C_on_dev;
}

struct square { __host__ __device__ float operator()(float x) { return x * x; } };
__host__ void Test_Matrix_F_Norm_On_GPU_Thrust(const Matrix& A,/*result*/float& norm)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;
	
	thrust::device_vector<float> d_vec(A.elements_on_host.begin(), A.elements_on_host.end());
	// cout << "Size: " << A.elements_on_host.size() << endl;

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel
	norm = sqrt(thrust::transform_reduce(d_vec.begin(), d_vec.end(), square(), 0.0f, thrust::plus<float>()));

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	// cout << "Thrust result " << norm << endl;
	printf("\nGPU runtime for F norm of using Thrust: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

__host__ void Test_Matrix_F_Norm_On_GPU(const Matrix& A,/*result*/float& norm)
{
	//// Load A and B to device memory
	Matrix A_on_dev(A.m,A.n,false);
	A_on_dev=A;

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//// Invoke kernel

	int maxThreads = 256; 
    int maxBlocks = maxThreads * 2; //1024;
	int n = A.elements_on_host.size();
	float * a_in = A_on_dev.elements_on_dev;
	norm = sqrt(runReduce(n, maxThreads, maxBlocks, a_in)); 

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime for F norm: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

void CHECK_RESULT(const Matrix & A, int ans, string info, bool if_break = true) {
	bool flag = false;
	for (int i = 0; i < A.m; i++) {
		for (int j = 0; j < A.n; j++) {
			if (A(i, j) != ans) {
				cout << "[WRONG RESULT]  " << info << "  " <<  i << " " << j << " " << A(i, j) << endl;
				flag = true; 
			} if (if_break && flag) break;
		} if (if_break && flag) break;
	}
}

int main()
{
	const int m=1024;
	const int n=4096;
	const int p=1024;

	Matrix h_A(m,n);
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			h_A(i,j)=1.0; // i*h_A.m + j; for testing transpose
		}
	}

	// Matrix h_AT(n, m); Test_Transpose_On_GPU(h_A, h_AT); return 0;

	Matrix h_B(n,p);
	for(int i=0;i<n;i++){
		for(int j=0;j<p;j++){
			h_B(i,j)=1.f;
		}
	}

	Matrix h_B2(m,m);	////for testing A^TBA
	for(int i=0;i<m;i++){
		for(int j=0;j<m;j++){
			h_B2(i,j)=1.f;
		}
	}


	Matrix h_C(m,p);
	Matrix h_C2(n,n);

	Test_Matrix_Multiplication_AB_On_GPU(h_A,h_B,h_C);
	cout<<"AB result: "<<h_C(h_C.m/2,h_C.n/2)<<endl;

	// CHECK_RESULT(h_C, h_A.n, "AB");
	
	bool testTwiceMultiplication = false;
	if (testTwiceMultiplication) { 
		Test_Matrix_Multiplication_ATBA_On_GPU_TwiceMultiplication(h_A,h_B2,h_C2);
		cout << "Test_Matrix_Multiplication_ATBA_On_GPU_TwiceMultiplication" << endl;
	} else {
		Test_Matrix_Multiplication_ATBA_On_GPU(h_A,h_B2,h_C2);
	}
	cout<<"ATBA result: "<<h_C2(h_C2.m/3,h_C2.n/3)<<endl;

	// CHECK_RESULT(h_C2, h_A.m*h_A.m, "ATBA");
	
	float f_norm=0.f;
	bool testThrust = false;
	if (testThrust){
		Test_Matrix_F_Norm_On_GPU_Thrust(h_A,f_norm);
		cout << "Calculating F-norm using Thrust" << endl;
	} else {
		Test_Matrix_F_Norm_On_GPU(h_A,f_norm);
	}
	cout<<"F-norm result: "<<f_norm<<endl;

	return 0;
}