# Fourth homework assignment

## Task: Implementing Matrix multiplication using 

In this task, you will implement matrix multiplication using the WMMA API and TensorCore, employing a block-based approach to improve performance. Begin by partitioning the input matrices into blocks that fit into the GPU's shared memory, leveraging TensorCore to accelerate the multiplication within each block. Implement the CUDA kernel (mm_block_tc) to handle block-wise matrix multiplication efficiently. Then, compare the execution time of your TensorCore-accelerated kernel against a naive matrix multiplication (mm_naive) implementation for varying sizes of square matrices (256, 512, 1024, 2048, and 4096).

### Analysis and Reporting:
Describe your results in a report (two-page max).

