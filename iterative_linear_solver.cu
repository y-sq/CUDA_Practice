//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 3: sparse linear solver
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
  
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


using namespace std;


const int n=128;							////grid size, we will change this value to up to 256 to test your code
const int g=1;							////padding size
const int s=(n+2*g)*(n+2*g);			////array size
#define I(i,j) ((i+g)*(n+2*g)+(j+g))		////2D coordinate -> array index
#define B(i,j) (i<0||i>=n||j<0||j>=n)		////check boundary
const bool verbose=false;				////set false to turn off print for x and residual
const double tolerance=1e-3;			////tolerance for the iterative solver

template <bool Red>
__global__ void Simple_Red_Black(double* x, const double* b) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int j = blockIdx.y * blockDim.y * 2;
	if (Red) { j += 2*threadIdx.y+(threadIdx.x&1); }
	else { j += 2*threadIdx.y+1-(threadIdx.x&1); }
	x[I(i, j)] = (b[I(i,j)]+x[I(i-1,j)]+x[I(i+1,j)]+x[I(i,j-1)]+x[I(i,j+1)])/4.0;
}

template <int BlockDim_X, int BlockDim_Y>
__global__ void Red_Gauss_Seidel_Solver(double* x, const double* b) 
{
	constexpr int height = BlockDim_X+2;
	constexpr int width = BlockDim_Y+1;
	__shared__ double arr_shared[height][width];
	int start_x = blockIdx.x * BlockDim_X;
	int start_y = blockIdx.y * BlockDim_Y * 2;
	int i = threadIdx.x; 
	int j = 2*threadIdx.y+(threadIdx.x&1);

	arr_shared[threadIdx.x][threadIdx.y] = x[I(start_x+i-1, start_y+j-2*(threadIdx.x&1))];
	if (threadIdx.x < 2) {
		arr_shared[threadIdx.x + BlockDim_X][threadIdx.y+(threadIdx.x&1)] = x[I(start_x+i-1+BlockDim_X, start_y+j)];
	} 
	if (threadIdx.y < 1) {
		arr_shared[threadIdx.x+1][threadIdx.y + BlockDim_Y] = x[I(start_x+i, start_y+2*BlockDim_Y+j-1)];
	}

	__syncthreads();
	x[I(start_x+i, start_y+j)] = (b[I(start_x+i, start_y+j)] 
								+ arr_shared[threadIdx.x][threadIdx.y + (threadIdx.x&1)] 
								+ arr_shared[threadIdx.x+1][threadIdx.y]
								+ arr_shared[threadIdx.x+1][threadIdx.y+1]
								+ arr_shared[threadIdx.x+2][threadIdx.y + (threadIdx.x&1)]) / 4.0;
}


template <int BlockDim_X, int BlockDim_Y>
__global__ void Black_Gauss_Seidel_Solver(double* x, const double* b) 
{
	constexpr int height = BlockDim_X+2;
	constexpr int width = BlockDim_Y+1;
	__shared__ double arr_shared[height][width];
	int start_x = blockIdx.x * BlockDim_X;
	int start_y = blockIdx.y * BlockDim_Y * 2;
	int i = threadIdx.x; 
	int j = 2*threadIdx.y+1-(threadIdx.x&1);

	arr_shared[threadIdx.x][threadIdx.y+1] = x[I(start_x+i-1, start_y+j+2*((threadIdx.x&1)))];
	if (threadIdx.x < 2) {
		arr_shared[threadIdx.x + BlockDim_X][threadIdx.y+1-(threadIdx.x&1)] = x[I(start_x+i-1+BlockDim_X, start_y+j)];
	} 
	if (threadIdx.y < 1) {
		arr_shared[threadIdx.x+1][threadIdx.y] = x[I(start_x+i, start_y+j-1)];
	}

	__syncthreads();
	x[I(start_x+i, start_y+j)] = (b[I(start_x+i, start_y+j)] 
								+ arr_shared[threadIdx.x][threadIdx.y + 1-(threadIdx.x&1)] 
								+ arr_shared[threadIdx.x+1][threadIdx.y]
								+ arr_shared[threadIdx.x+1][threadIdx.y+1]
								+ arr_shared[threadIdx.x+2][threadIdx.y + 1-(threadIdx.x&1)]) / 4.0;
}

__global__ void Compute_Res(const double* x, const double* b, double* res, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	res[i*n+j] = 4.0*x[I(i,j)]-x[I(i-1,j)]-x[I(i+1,j)]-x[I(i,j-1)]-x[I(i,j+1)]-b[I(i,j)];
}  

// kernel to cal res


//////////////////////////////////////////////////////////////////////////
////TODO 1: your GPU variables and functions start here

////Your implementations end here
//////////////////////////////////////////////////////////////////////////

template <typename T> 
inline __host__ T* copyMemToGpu(T* host_arr, size_t s, bool if_copy = true)
{
	T* dev_arr;
	cudaMalloc((void**)&dev_arr, s*sizeof(T));
	if (if_copy) cudaMemcpy(dev_arr, host_arr, s*sizeof(T), cudaMemcpyHostToDevice);
	return dev_arr;
} 

ofstream out;

//////////////////////////////////////////////////////////////////////////
////GPU test function
void Test_GPU_Solver()
{
	double* x=new double[s];
	memset(x,0x0000,sizeof(double)*s);
	double* b=new double[s];

	//////////////////////////////////////////////////////////////////////////
	////initialize x and b
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			b[I(i,j)]=4.0;		////set the values for the right-hand side
		}
	}
	for(int i=-1;i<=n;i++){
		for(int j=-1;j<=n;j++){
			if(B(i,j))
				x[I(i,j)]=(double)(i*i+j*j);	////set boundary condition for x
		}
	}

	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	//////////////////////////////////////////////////////////////////////////
	////TODO 2: call your GPU functions here
	////Requirement: You need to copy data from the CPU arrays, conduct computations on the GPU, and copy the values back from GPU to CPU
	////The final positions should be stored in the same place as the CPU function, i.e., the array of x
	////The correctness of your simulation will be evaluated by the residual (<1e-3)
	//////////////////////////////////////////////////////////////////////////

	thrust::device_vector<double> x_dev_vec(x, x+s);
	double* dev_x = thrust::raw_pointer_cast( &x_dev_vec[0] );	
	thrust::device_vector<double> b_dev_vec(b, b+s);
	double* dev_b = thrust::raw_pointer_cast( &b_dev_vec[0] );

	thrust::device_vector<double> res_vec_dev(n*n, 0.0);
	double* dev_res = thrust::raw_pointer_cast( &res_vec_dev[0] );

	// double* dev_x = copyMemToGpu(x, s);
	// double* dev_b = copyMemToGpu(b, s);
	
	const int block_x=8;
	const int block_y=32;
	const int grid_x = n / block_x;
	const int grid_y = n / (block_y*2);
	
	auto square = [=] __device__ (double x) {return x*x;};
	int iter_num = 0;
	while (true) {
		// Simple_Red_Black<true> <<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>>(dev_x, dev_b);
		// Simple_Red_Black<false> <<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>>(dev_x, dev_b);
		Red_Gauss_Seidel_Solver<block_x, block_y> <<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>> (dev_x, dev_b);
		Black_Gauss_Seidel_Solver<block_x, block_y> <<<dim3(grid_x, grid_y), dim3(block_x, block_y)>>> (dev_x, dev_b);
		iter_num++; 
		if ((iter_num & 0x3f) == 0) {
			Compute_Res<<<dim3(grid_x, grid_y*2), dim3(block_x, block_y)>>> (dev_x, dev_b, dev_res, n);
			double res = thrust::transform_reduce(res_vec_dev.begin(), res_vec_dev.end(), square, 0.0f, thrust::plus<double>());
			// cout << res << endl; break;
			if (res <= tolerance) { break; }
		}	
	} 


	cudaMemcpy(x, dev_x, s*sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(b, dev_b, s*sizeof(double), cudaMemcpyDeviceToHost);
		
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	printf("\nGPU runtime: %.4f ms\n",gpu_time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////

	////output x
	if(verbose){
		cout<<"\n\nx for your GPU solver:\n";
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				cout<<x[I(i,j)]<<", ";
			}
			cout<<std::endl;
		}	
	}

	////calculate residual
	double residual=0.0;
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			residual+=pow(4.0*x[I(i,j)]-x[I(i-1,j)]-x[I(i+1,j)]-x[I(i,j-1)]-x[I(i,j+1)]-b[I(i,j)],2);
		}
	}
	cout<<"\n\nresidual for your GPU solver: "<<residual<<endl;

	out<<"R0: "<<residual<<endl;
	out<<"T1: "<<gpu_time<<endl;

	//////////////////////////////////////////////////////////////////////////

	delete [] x;
	delete [] b;
}

int main()
{
	
	// int devicesCount;
    // cudaGetDeviceCount(&devicesCount);
	// cudaSetDevice(devicesCount-1);
	
	Test_GPU_Solver();	////Test function for your own GPU implementation

	return 0;
}