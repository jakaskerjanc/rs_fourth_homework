#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#define BLOCKSIZE 16
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define WARP_SIZE 32

using namespace nvcuda;

void init_matrices(half *a, half *b, float *c, int matsize) {
    for (int i = 0; i < matsize; ++i) {
        for (int j = 0; j < matsize; ++j) {
            a[i * matsize + j] = __float2half(1.0);
            b[i * matsize + j] = __float2half(1.0);
            c[i * matsize + j] = 1.0;
        }
    }
}

__global__ void mm_block_tc(int mat_size, half *A, half *B, float *C) {
    // Tile using a 2D grid
    int warpX = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpY = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, float> acc_frag;
    wmma::fragment<wmma::accumulator, BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the K dimension
    for (int i = 0; i < mat_size; i += BLOCKSIZE) {
        // Bounds checking
        int aRow = warpX * BLOCKSIZE;
        int aCol = i;
        int bRow = i;
        int bCol = warpY * BLOCKSIZE;

        // Load the inputs
        wmma::load_matrix_sync(a_frag, A + aRow * mat_size + aCol, mat_size);
        wmma::load_matrix_sync(b_frag, B + bRow * mat_size + bCol, mat_size);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpX * BLOCKSIZE;
    int cCol = warpY * BLOCKSIZE;

    wmma::load_matrix_sync(c_frag, C + cRow * mat_size + cCol, mat_size, wmma::mem_row_major);

    for(int i=0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = acc_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(C + cRow * mat_size + cCol, c_frag, mat_size, wmma::mem_row_major);
}


__global__ void mm_naive(int mat_size,  half *A, half *B, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < mat_size && y < mat_size) {
    float tmp = 0.0;
    for (int i = 0; i < mat_size; ++i) {
      tmp += (float)A[x * mat_size + i] * (float)B[i * mat_size + y];
    }

    C[x * mat_size + y] = tmp ;
  }
}

void run_matrix_multiplication(int N) {
    printf("\nRunning matrix multiplication for size %d x %d\n", N, N);
    
    half *mat_a, *mat_b;
    float *mat_c;

    mat_a = (half*)malloc(N * N * sizeof(half));
    mat_b = (half*)malloc(N * N * sizeof(half));
    mat_c = (float*)malloc(N * N * sizeof(float));

    init_matrices(mat_a, mat_b, mat_c, N);

    half *d_mat_a, *d_mat_b;
    float *d_mat_c;
    cudaMalloc(&d_mat_a, N * N * sizeof(half));
    cudaMalloc(&d_mat_b, N * N * sizeof(half));
    cudaMalloc(&d_mat_c, N * N * sizeof(float));

    cudaMemcpy(d_mat_a, mat_a, N * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_b, mat_b, N * N * sizeof(half), cudaMemcpyHostToDevice);
   
    dim3 gridDim;
    dim3 blockDim; 
    blockDim.x = 32;
    blockDim.y = 1;
    gridDim.x = CEIL_DIV(N, BLOCKSIZE * blockDim.x / 32); 
    gridDim.y = CEIL_DIV(N, BLOCKSIZE * blockDim.y);

    cudaEvent_t start1, stop1, start2, stop2;
    float milliseconds = 0;

    // Warm up
    mm_naive<<<gridDim, blockDim>>>(N, d_mat_a, d_mat_b, d_mat_c);
    cudaDeviceSynchronize();

    // TensorCore version
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    mm_block_tc<<<gridDim, blockDim>>>(N, d_mat_a, d_mat_b, d_mat_c);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&milliseconds, start2, stop2);
    printf("Time taken with TensorCore: %f ms\n", milliseconds);

    // Naive version
    blockDim.x = 32;
    blockDim.y = 1;
    gridDim.x = CEIL_DIV(N, blockDim.x / 32); 
    gridDim.y = CEIL_DIV(N, blockDim.y);

    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    mm_naive<<<gridDim, blockDim>>>(N, d_mat_a, d_mat_b, d_mat_c);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("Time taken without TensorCore: %f ms\n", milliseconds);
    
    cudaMemcpy(mat_c, d_mat_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);  

    free(mat_a);
    free(mat_b);
    free(mat_c);
    cudaFree(d_mat_a);
    cudaFree(d_mat_b);
    cudaFree(d_mat_c);
}

int main() {
    int sizes[] = {256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        run_matrix_multiplication(sizes[i]);
        run_matrix_multiplication(sizes[i]);
        run_matrix_multiplication(sizes[i]);
        run_matrix_multiplication(sizes[i]);
        run_matrix_multiplication(sizes[i]);
    }
    
    return 0;
}