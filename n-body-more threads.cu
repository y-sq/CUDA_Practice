//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round 2: n-body simulation
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error
__device__ void atomicAdd_(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    // return __longlong_as_double(old);
}

template <int THREAD_NUM, int ROW_NUM, bool IS_MULTIPLE>  // int BLOCK_NUM, 
__global__ void N_Body_Simulation(	double* pos_x,double* pos_y,double* pos_z,		////position array
									double* vel_x,double* vel_y,double* vel_z,		////velocity array
									double* new_pos_x,double* new_pos_y,double* new_pos_z,		
									const double* mass,								////mass array
									const int P_N,									////number of particles
									const double DT,								////timestep
									const double EPS_SQ								////epsilon to avoid 0-denominator
								)					
{
	__shared__ double smem_x[THREAD_NUM];
	__shared__ double smem_y[THREAD_NUM];
	__shared__ double smem_z[THREAD_NUM];
	__shared__ double smem_m[THREAD_NUM];

	int idx_s = threadIdx.x;
	int idx_local = idx_s & (ROW_NUM-1);
	int idx = blockIdx.x * ROW_NUM + idx_local;

	double this_x, this_y, this_z;
	double diff_x, diff_y, diff_z;
	double squared, inv_square_root, coef;
	double this_acl_x = 0.0, this_acl_y = 0.0, this_acl_z = 0.0; 

	if (IS_MULTIPLE || idx < P_N) {
		this_x = pos_x[idx];
		this_y = pos_y[idx];
		this_z = pos_z[idx];
	}

	#pragma unroll (4)
	for (int start = 0; start < P_N; start += THREAD_NUM) { 
		int idx_t = start + idx_s; 
		if (IS_MULTIPLE || idx_t < P_N) { 
			smem_x[idx_s] = pos_x[idx_t];
			smem_y[idx_s] = pos_y[idx_t];
			smem_z[idx_s] = pos_z[idx_t];
			smem_m[idx_s] = mass[idx_t];
		}
		__syncthreads();
		int start_i = idx_s - idx_local; 
		#pragma unroll
		for (int i = 0; i < ROW_NUM; ++i) {
			if (!IS_MULTIPLE && start + start_i + i >= P_N) { break; }
			diff_x = smem_x[start_i+i] - this_x;
			diff_y = smem_y[start_i+i] - this_y;
			diff_z = smem_z[start_i+i] - this_z;
			squared = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
			inv_square_root = rsqrt(squared + EPS_SQ);
			coef = inv_square_root * inv_square_root * inv_square_root * smem_m[start_i+i];
			this_acl_x += coef * diff_x;
			this_acl_y += coef * diff_y;
			this_acl_z += coef * diff_z;
		}
		__syncthreads();
	}
	if (idx_s < ROW_NUM) { smem_x[idx_s] = 0.0f; smem_y[idx_s] = 0.0f; smem_z[idx_s] = 0.0f; }
	__syncthreads();
	atomicAdd_(&smem_x[idx_local], this_acl_x);
	atomicAdd_(&smem_y[idx_local], this_acl_y);
	atomicAdd_(&smem_z[idx_local], this_acl_z);
	__syncthreads();

	if ((!IS_MULTIPLE && idx >= P_N) || idx_s >= ROW_NUM) { return; }
	if (idx_s < ROW_NUM) {
		double this_vel_x = (vel_x[idx] += smem_x[idx_local] * DT);
		double this_vel_y = (vel_y[idx] += smem_y[idx_local] * DT);
		double this_vel_z = (vel_z[idx] += smem_z[idx_local] * DT);
		new_pos_x[idx] = this_x + this_vel_x * DT;
		new_pos_y[idx] = this_y + this_vel_y * DT;
		new_pos_z[idx] = this_z + this_vel_z * DT;
	}
}



const double dt=0.001;							////time step
const int time_step_num=10;						////number of time steps
const double epsilon=1e-2;						////epsilon added in the denominator to avoid 0-division when calculating the gravitational force
const double epsilon_squared=epsilon*epsilon;	////epsilon squared

const unsigned int grid_size=16;					////assuming particles are initialized on a background grid
const unsigned int particle_n=pow(grid_size,3);	////assuming each grid cell has one particle at the beginning

template <typename T> 
inline __host__ T* copyMemToGpu(T* host_arr, size_t s, bool if_copy = true)
{
	T* dev_arr;
	cudaMalloc((void**)&dev_arr, s*sizeof(T));
	if (if_copy) cudaMemcpy(dev_arr, host_arr, s*sizeof(T), cudaMemcpyHostToDevice);
	return dev_arr;
} 

__host__ void Test_N_Body_Simulation()
{
	////initialize position, velocity, acceleration, and mass
	
	double* pos_x=new double[particle_n];
	double* pos_y=new double[particle_n];
	double* pos_z=new double[particle_n];
	////initialize particle positions as the cell centers on a background grid
	double dx=1.0/(double)grid_size;
	for(unsigned int k=0;k<grid_size;k++){
		for(unsigned int j=0;j<grid_size;j++){
			for(unsigned int i=0;i<grid_size;i++){
				unsigned int index=k*grid_size*grid_size+j*grid_size+i;
				pos_x[index]=dx*(double)i;
				pos_y[index]=dx*(double)j;
				pos_z[index]=dx*(double)k;
			}
		}
	}

	double* vel_x=new double[particle_n];
	memset(vel_x,0x00,particle_n*sizeof(double));
	double* vel_y=new double[particle_n];
	memset(vel_y,0x00,particle_n*sizeof(double));
	double* vel_z=new double[particle_n];
	memset(vel_z,0x00,particle_n*sizeof(double));

	double* acl_x=new double[particle_n];
	memset(acl_x,0x00,particle_n*sizeof(double));
	double* acl_y=new double[particle_n];
	memset(acl_y,0x00,particle_n*sizeof(double));
	double* acl_z=new double[particle_n];
	memset(acl_z,0x00,particle_n*sizeof(double));

	double* mass=new double[particle_n];
	for(int i=0;i<particle_n;i++){
		mass[i]=100.0;
	}


	cudaEvent_t start,end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	float gpu_time=0.0f;
	cudaDeviceSynchronize();
	cudaEventRecord(start);

	double* dev_pos_x = copyMemToGpu(pos_x, particle_n);
	double* dev_pos_y = copyMemToGpu(pos_y, particle_n);
	double* dev_pos_z = copyMemToGpu(pos_z, particle_n);
	double* dev_new_pos_x = copyMemToGpu(pos_x, particle_n, false);
	double* dev_new_pos_y = copyMemToGpu(pos_y, particle_n, false);
	double* dev_new_pos_z = copyMemToGpu(pos_z, particle_n, false);
	double* dev_vel_x = copyMemToGpu(vel_x, particle_n);
	double* dev_vel_y = copyMemToGpu(vel_y, particle_n);
	double* dev_vel_z = copyMemToGpu(vel_z, particle_n);
	double* dev_mass  = copyMemToGpu(mass, particle_n);

	const int ROW_NUM = 64;
	const int THREAD_NUM = ROW_NUM << 2; 
	const int BLOCK_NUM = (particle_n-1) / ROW_NUM + 1;
	bool is_multiple = ((particle_n % THREAD_NUM) == 0);
	// cout << THREAD_NUM << " " << BLOCK_NUM << " " << is_multiple << endl;

	for(int i=0;i<time_step_num;i++){
		if (is_multiple) {
			N_Body_Simulation<THREAD_NUM, ROW_NUM, true><<<BLOCK_NUM, THREAD_NUM>>>
			    (  dev_pos_x, dev_pos_y, dev_pos_z,
				   dev_vel_x, dev_vel_y, dev_vel_z,
				   dev_new_pos_x, dev_new_pos_y, dev_new_pos_z,
				   dev_mass,
				   particle_n , dt, //epsilon, 
				   epsilon_squared
				);
		} else {
			N_Body_Simulation<THREAD_NUM, ROW_NUM, false><<<BLOCK_NUM, THREAD_NUM>>>
				(  dev_pos_x, dev_pos_y, dev_pos_z,
					dev_vel_x, dev_vel_y, dev_vel_z,
					dev_new_pos_x, dev_new_pos_y, dev_new_pos_z,
					dev_mass,
					particle_n, dt, //epsilon, 
					epsilon_squared
				);
		} 
		swap(dev_pos_x, dev_new_pos_x);
		swap(dev_pos_y, dev_new_pos_y);
		swap(dev_pos_z, dev_new_pos_z);
	}

	cudaMemcpy(pos_x, dev_pos_x, particle_n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(pos_y, dev_pos_y, particle_n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(pos_z, dev_pos_z, particle_n*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_pos_x);
	cudaFree(dev_pos_y);
	cudaFree(dev_pos_z);
	cudaFree(dev_new_pos_x);
	cudaFree(dev_new_pos_y);
	cudaFree(dev_new_pos_z);
	cudaFree(dev_vel_x);
	cudaFree(dev_vel_y);
	cudaFree(dev_vel_z);
	cudaFree(dev_mass);


	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time,start,end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	//////////////////////////////////////////////////////////////////////////

	cout<<"R0: "<<pos_x[particle_n/2]<<" " <<pos_y[particle_n/2]<<" " <<pos_z[particle_n/2]<<endl;
	cout<<"T1: "<<gpu_time<<endl;
}

int main()
{

	int devicesCount;
    cudaGetDeviceCount(&devicesCount);
	cudaSetDevice(devicesCount-1);
	// cout << devicesCount << endl;
	// while (true)
	Test_N_Body_Simulation();

	return 0;
}