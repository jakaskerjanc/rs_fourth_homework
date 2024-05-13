# Fourth homework assignment

## Task: Implementing Matrix multiplication using 

In this task, you will implement matrix multiplication using the WMMA PTX API and TensorCore, employing a block-based approach to improve performance. First, partition the input matrices into blocks and leverage TensorCore to accelerate block matrix multiplication. Implement the CUDA kernel (mm_block_tc) to handle block-wise matrix multiplication efficiently. Then, compare the execution time of your TensorCore-accelerated kernel against a naive matrix multiplication (mm_naive) implementation for varying sizes of square matrices (256, 512, 1024, 2048, and 4096). In the given program, the block size is equal to the number of threads in a warp; you can change it to suit your needs. To enable fair comparison, one should compare the performance of the naive kernel and the TensorCore accelerated kernel under the same grid and block size configuration. 

### Analysis and Reporting:
Describe your results in a report (two pages max). Note: The employment of WMMA C/C++ API is not prohibited; however, do not expect support when problems arise. Strongly recommend PTX. 
