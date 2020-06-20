//////////////////////////////////////////////////////////////////////////
////This is the code implementation for GPU Premier League Round Final: conjugate gradient solver
//////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>

// #define PRE_COND
// #define DEBUG
#define GRID_SIZE 256

const int grid_size=GRID_SIZE;										////grid size, we will change this value to up to 256 to test your code, notice that we do not have padding elements
const int s=grid_size*grid_size;								////array size
#define I(i,j) ((i)*grid_size+(j))								////2D coordinate -> array index
#define B(i,j) (i)<0||(i)>=grid_size||(j)<0||(j)>=grid_size		////check boundary
const bool verbose=false;										////set false to turn off print for x and residual
const int max_iter_num=1000;									////max cg iteration number
const double tolerance=1e-3;									////tolerance for the iterative solver

void MV(/*CRS sparse matrix*/const double* val,const int* col,const int* ptr,/*number of column*/const int n,/*input vector*/const double* v,/*result*/double* mv) {
    for (int row = 0; row < n; row++) {
        double tmp = 0.;
        for (int j = ptr[row]; j < ptr[row+1]; j++) { // j is the index of the val which belongs to current row
            tmp += val[j] * v[col[j]];
        }
        mv[row] = tmp;
    }
}

////return the dot product between a and b
double Dot(const double* a,const double* b,const int n) {
    double res = 0.0;
    for (int i = 0; i < n; i++) {
        res += a[i] * b[i];
    }
    return res;
}

/*
struct my_func_mv {
    const thrust::device_vector<double>& v;
    const thrust::device_vector<int>& col;
    const thrust::device_vector<double>& val;
    my_func_mv(const thrust::device_vector<double> & _v,
        const thrust::device_vector<int> & _col,
        const thrust::device_vector<double> & _val): 
        v(_v), col(_col), val(_val) {}
    __host__ __device__ double operator()(const int & i1, const int & i2) const {
        double tmp = 0.0;
        for (int j = i1; j < i2; j++) {
            tmp += v[col[j]] * val[j];
        }    
        return tmp;
    }
};
void MV_GPU3(const thrust::device_vector<double>& val, const thrust::device_vector<int>& col, const thrust::device_vector<int>& ptr,
    const int n, const thrust::device_vector<double>& v, thrust::device_vector<double>& mv) {
    thrust::transform(ptr.begin(), ptr.begin()+n, ptr.begin()+1, mv.begin(), my_func_mv(v, col, val));
}
*/

__global__ void MV4cuda(/*CRS sparse matrix*/const double* val,const int* col,const int* ptr,/*number of column*/const int n,
        /*input vector*/const double* v,/*result*/double* mv) {
    
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = t_idx; i < n; i += blockDim.x * gridDim.x) {
        // int i = blockIdx.x * blockDim.x + threadIdx.x;
        double tmp = 0;
        int start = ptr[i], end = ptr[i + 1];
	    #pragma unroll (3)  // useful or not ?
        for (int j = start; j < end; j++) {
            tmp += val[j] * v[col[j]];
        }
        mv[i] = tmp; 
    }

}

double Dot_GPU(const thrust::device_vector<double>& d1, const thrust::device_vector<double>& d2) {
    return thrust::inner_product(d1.begin(), d1.end(), d2.begin(), (double) 0.0);
}

struct my_func_add {
    double a = 0.0;
    my_func_add(double _a) : a(_a) {}
    __host__ __device__ double operator()(const double &x, const double &y) const {
        return x + a * y;
    }
};

// customized transform, des=d1+a*d2
void Add_GPU(const thrust::device_vector<double>& d1, const thrust::device_vector<double>& d2, double alpha,
             thrust::device_vector<double>& des) {
    thrust::transform(d1.begin(), d1.end(), d2.begin(), des.begin(), my_func_add(alpha));
}

struct my_func_add_inplace {
    double a;
    double b;
    my_func_add_inplace(double _a, double _b) : a(_a), b(_b) {}
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<0>(t) = a * thrust::get<0>(t) + b * thrust::get<1>(t);
    }
}; // v1 <- alpha1 * v1 + alpha2 * v2
void Add_Inplace_GPU(thrust::device_vector<double>& v1, const thrust::device_vector<double>& v2, double alpha1, double alpha2) {
    // thrust::for_each (InputIterator first, InputIterator last, UnaryFunction f)
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(v1.begin(), v2.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(v1.end(),   v2.end())),
                     my_func_add_inplace(alpha1, alpha2));
} 
// https://stackoverflow.com/questions/7513476/stl-thrust-multiple-vector-transform

struct my_func_multi {
    my_func_multi(){}
    __host__ __device__ double operator()(const double &x, const double &y) const {
        return x * y;
    }
};
// v1 <- inv_diag r
void Apply_Inv_Diag(thrust::device_vector<double>& v1, const thrust::device_vector<double>& r, const thrust::device_vector<double>& inv_d) {
    thrust::transform(r.begin(), r.end(), inv_d.begin(), v1.begin(), my_func_multi());
}

__global__ void fill_inv_diag(/*CRS sparse matrix*/const double* val,const int* col,const int* ptr,/*number of column*/const int n,
    /*result*/double* diag) {
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = t_idx; i < n; i += blockDim.x * gridDim.x) {
        int start = ptr[i], end = ptr[i + 1];
        for (int j = start; j < end; j++) {
            if (col[j] == i) {
                diag[i] = 1.0 / val[j];
                return;
            }
        }
    }
}

void Conjugate_Gradient_Solver_GPU (
        std::vector<double>& val_host,
        const std::vector<int>& col_host,
        const std::vector<int>& ptr_host,
        const int n,
        std::vector<double>& r_host,
        std::vector<double>& q_host,
        std::vector<double>& d_host,
        std::vector<double>& x_host,
        const std::vector<double>& b_host,
        const int max_iter, const double tol) {


    // convert std::vector to thrust vectors
    thrust::device_vector<double> val = val_host;
    thrust::device_vector<int> col = col_host;
    thrust::device_vector<int> ptr = ptr_host;
    thrust::device_vector<double> b = b_host;
    thrust::device_vector<double> x(n, 0.);
    thrust::device_vector<double> r(n, 0.);
    thrust::device_vector<double> q(n, 0.);
    thrust::device_vector<double> d(n, 0.);

    // raw pointers
    double* dev_val = thrust::raw_pointer_cast( &val[0] );
    int* dev_col = thrust::raw_pointer_cast( &col[0] );
    int* dev_ptr = thrust::raw_pointer_cast( &ptr[0] );
    double* dev_b = thrust::raw_pointer_cast( &b[0] );
    double* dev_x = thrust::raw_pointer_cast( &x[0] );
    double* dev_r = thrust::raw_pointer_cast( &r[0] );
    double* dev_q = thrust::raw_pointer_cast( &q[0] );
    double* dev_d = thrust::raw_pointer_cast( &d[0] );


    // thread_num, block_num
    int thread_num = 128;
    int block_num = int(ceil(n/128.0));


#ifdef PRE_COND
    thrust::device_vector<double> s(n, 0.); 
    thrust::device_vector<double> inv_diag(n, 0.0);  // pre_cond specifically for 2D_Poisson_Problem
    fill_inv_diag<<<block_num, thread_num>>>(dev_val, dev_col, dev_ptr, n, thrust::raw_pointer_cast( &inv_diag[0] ));
#endif


    ////declare variables
    int iter = 0;
    double delta_old = 0.0;
    double delta_new = 0.0;
    double alpha = 0.0;
    double beta = 0.0;


    ////TODO: r=b-Ax
    MV4cuda<<<block_num, thread_num>>>(dev_val, dev_col, dev_ptr, n, dev_x, dev_r);
    Add_Inplace_GPU(r, b, -1, 1);
    
#ifndef PRE_COND
    ////TODO: d=r
    d = r; // dev_d = thrust::raw_pointer_cast( &d[0] );
    ////TODO: delta_new=rTr
    delta_new = Dot_GPU(r, r);
#else 
    ////TODO: d=inv_diag r 
    Apply_Inv_Diag(d, r, inv_diag);  // d <- M_{-1}r
    ////TODO: delta_new=rTd
    delta_new = Dot_GPU(d, r);
#endif

    ////Here we use the absolute tolerance instead of a relative one, which is slightly different from the notes
    while (iter < max_iter && delta_new > tol) {
        ////TODO: q=Ad
        MV4cuda<<<block_num, thread_num>>>(dev_val, dev_col, dev_ptr, n, dev_d, dev_q);
        ////TODO: alpha=delta_new/d^Tq
        alpha = delta_new / Dot_GPU(d, q);
    #ifdef DEBUG 
        std::cout << "GPU iter=" << iter << ", delta=" << delta_new << "\n"; // TODO DEBUG
    #endif
        ////TODO: x=x+alpha*d
        Add_Inplace_GPU(x, d, 1, alpha);
        if (iter % 50 == 0 && iter > 1) {
            ////TODO: r=b-Ax
            MV4cuda<<<block_num, thread_num>>>(dev_val, dev_col, dev_ptr, n, dev_x, dev_r);
            Add_Inplace_GPU(r, b, -1, 1);
        } else {
            ////TODO: r=r-alpha*q
            Add_Inplace_GPU(r, q, 1, -alpha);
        }
        ////TODO: delta_old=delta_new
        delta_old = delta_new;

    #ifndef PRE_COND
        ////TODO: delta_new=r^Tr
        delta_new = Dot_GPU(r, r);
        ////TODO: beta=delta_new/delta_old
        beta = delta_new / delta_old;
        ////TODO: d=r+beta*d 
        Add_Inplace_GPU(d, r, beta, 1);
    #else 
        ////TODO: s = M_{-1} r
        Apply_Inv_Diag(s, r, inv_diag); 
        ////TODO: delta_new = r dot s
        delta_new = Dot_GPU(r, s);
        ////TODO: beta=delta_new/delta_old
        beta = delta_new / delta_old;
        ////TODO: d=s+beta*d
        Add_Inplace_GPU(d, s, beta, 1);
    #endif

        ////TODO: increase the counter
        iter++;
    }
    if (iter < max_iter)
        std::cout << "GPU conjugate gradient solver converges after " << iter << " iterations with residual "
                  << (delta_new) << std::endl;
    else
        std::cout << "GPU conjugate gradient solver does not converge after " << max_iter
                  << " iterations with residual " << (delta_new) << std::endl;

    // thrust::device_vector to std::vector
    // thrust::copy(r.begin(), r.end(), r_host.begin());
    // thrust::copy(q.begin(), q.end(), q_host.begin());
    // thrust::copy(d.begin(), d.end(), d_host.begin());
    thrust::copy(x.begin(), x.end(), x_host.begin());
}
//////////////////////////////////////////////////////////////////////////




std::ofstream out;

void Initialize_2D_Poisson_Problem(std::vector<double>& val,std::vector<int>& col,std::vector<int>& ptr,std::vector<double>& b) {
    ////assemble the CRS sparse matrix
    ////The grid dimension is grid_size x grid_size.
    ////The matrix's dimension is s x s, with s= grid_size*grid_size.
    ////We also initialize the right-hand vector b

    val.clear();
    col.clear();
    ptr.resize(s + 1, 0);
    b.resize(s, -4.);

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            int r = I(i, j);
            int nnz_for_row_r = 0;

            ////set (i,j-1)
            if (!(B(i, j - 1))) {
                int c = I(i, j - 1);
                val.push_back(-1.);
                col.push_back(c);
                nnz_for_row_r++;
            } else {
                double boundary_val = (double) (i * i + (j - 1) * (j - 1));
                b[r] += boundary_val;
            }

            ////set (i-1,j)
            if (!(B(i - 1, j))) {
                int c = I(i - 1, j);
                val.push_back(-1.);
                col.push_back(c);
                nnz_for_row_r++;
            } else {
                double boundary_val = (double) ((i - 1) * (i - 1) + j * j);
                b[r] += boundary_val;
            }

            ////set (i+1,j)
            if (!(B(i + 1, j))) {
                int c = I(i + 1, j);
                val.push_back(-1.);
                col.push_back(c);
                nnz_for_row_r++;
            } else {
                double boundary_val = (double) ((i + 1) * (i + 1) + j * j);
                b[r] += boundary_val;
            }

            ////set (i,j+1)
            if (!(B(i, j + 1))) {
                int c = I(i, j + 1);
                val.push_back(-1.);
                col.push_back(c);
                nnz_for_row_r++;
            } else {
                double boundary_val = (double) (i * i + (j + 1) * (j + 1));
                b[r] += boundary_val;
            }

            ////set (i,j)
            {
                val.push_back(4.);
                col.push_back(r);
                nnz_for_row_r++;
            }
            ptr[r + 1] = ptr[r] + nnz_for_row_r;
        }
    }
}

void Test_GPU_Solver() {
    std::vector<double> val;
    std::vector<int> col;
    std::vector<int> ptr;
    std::vector<double> b;
    Initialize_2D_Poisson_Problem(val, col, ptr, b);

    std::vector<double> x(s, 0.);
    std::vector<double> r(s, 0.);
    std::vector<double> q(s, 0.);
    std::vector<double> d(s, 0.);


    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float gpu_time = 0.0f;
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    // call GPU function
    Conjugate_Gradient_Solver_GPU(val, col, ptr, s,
            r, q, d,
            x, b,
            max_iter_num, tolerance);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_time, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    //////////////////////////////////////////////////////////////////////////

    if (verbose) {
        std::cout << "\n\nx for CG on GPU:\n";
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++)
                std::cout << x[i * grid_size + j] << ", ";
            std::cout << "\n";
        }
    }
    std::cout << "\n\n";

    //////calculate residual
    MV(&val[0], &col[0], &ptr[0], s, &x[0], &r[0]);
    for (int i = 0; i < s; i++)r[i] = b[i] - r[i];
    double residual = Dot(&r[0], &r[0], s);
    std::cout << "\nGPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Residual for your GPU solver: " << residual << std::endl;

    out << "R1: " << residual << std::endl;
    out << "T1: " << gpu_time << std::endl;
}

int main() {
    
    Test_GPU_Solver();

    return 0;
}

