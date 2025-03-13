# Chapter 4 Exercises

## Exercise 4.1
Consider the following CUDA kernel and the corresponding host function that calls it:
```cpp
01 __global__ void foo_kernel(int* a, int* b) {
02     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03     if(threadIdx.x < 40 || threadIdx.x >= 104) {
04         b[i] = a[i] + 1;
05     }
06     if(i%2 == 0) {
07         a[i] = b[i]*2;
08     }
09     for(unsigned int j = 0; j < 5 - (i%3); ++j) {
10         b[i] += j;
11     }
12 }
13 void foo(int* a_d, int* b_d) {
14     unsigned int N = 1024;
15     foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
16 }
```
- a. What is number of warps per block?
  
  - **Ans**: 128/32 = 4 warps per block
    
- b. What is the number of warps in the grid?

  - **Ans**: The total number of blocks in the grid is 8 blocks; thus, the total number of warps in the grid is 4 * 8 = 32 warps.
    
- c. For the statement on line 04:
  -  i. How many warps in the grid are active?
  
    -  **Ans**: For condition `threadIdx.x < 40`, 0-39 threads are active, which means there are 2 warps (0-31, 31-39) are active; For condition `threadIdx.x >= 104`, 104-127 threads are active, which means there is only one warp (104-127) is active
Thus total 3 warps are active in a block, for 8 blocks, the total number of warps active is 24.

  - ii. How many warps are divergent in the grid?
 
    -  **Ans**: Warps are divergent if threads in the warp take different paths. From question i, there are 2 warps that are partially active, thus, 16 warps are divergent in the grid.
   
  - iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
 
    - **Ans**: SIMD efficiency is the ratio of active threads to the total number of threads in the warp. For warp 0 of bock 0, all threads are active in the warp 0, thus the SIMD efficiency (in %) of warp 0 of block 0 is 100%
   
  - iv. What is the SIMD efficiency (in %) of warp 1 of block 0?
 
    - **Ans**:  For warp 1 of block 0, the total number of active threads is 8, thus the SIMD efficiency (in %) of warp 0 of block 0 is 8 / 32 = 25%
   
  - v. What is the SIMD efficiency (in %) of warp 3 of block 0?
 
    - **Ans**:  For warp 3 of block 1, the total number of active threads is 24, thus the SIMD efficiency (in %) of warp 3 of block 0 is 24 / 32 = 75%
   
- d. For the statement on line 07:
  - i. How many warps in the grid are active?
 
      - **Ans**: All warps in the grid are active. Thus, 32 warps are active in the grid.
   
  - ii. How many warps in the grid are divergent?
 
      - **Ans**:  All warps in the grid are divergent because only an even number of threads in the if path. Thus, 32 warps are divergent in the grid.
   
  - iii. What is the SIMD efficiency (in %) of warp 0 of block 0?
 
      - **Ans**:  For warp 0 of block 0, only half the threads are active. Thus the SIMD efficiency (in %) of warp 0 of block 0 is 50%
   
- e. For the statement on line 09:
    - i.  How many iterations have no divergence?
 
        -  **Ans**: The loop will run a minimum of 3 iterations and a maximum of 5 iterations. Three iterations have no divergence.
     
    - ii. How many iterations have divergence?
 
        - **Ans**: Two iterations have divergence.
     
## Exercise 4.2
For a vector addition, assume that the vector length is 2000, each thread
calculates one output element, and the thread block size is 512 threads. How
many threads will be in the grid?

- **Ans**: The total number of blocks are (2000 + 512 - 1) / 512 = 4. Therefore, 512 * 4 = 2048 threads in the grid.

## Exercise 4.3
For the previous question, how many warps do you expect to have divergence
due to the boundary check on vector length?

- **Ans**: Only one warp will have divergence.

## Exercise 4.4
Consider a hypothetical block with 8 threads executing a section of code
before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads’ total execution time is spent waiting for the barrier?

**Ans**: 
- Total time spent waiting for barrier = 1.0 + 0.7+ 0.2 + 0.6 + 1.1 + 0.4 + 0.1 = 4.1;
- Total time spent by threads is 3.0 * 8 = 24.0
- (4.1 / 24.0) * 100% = 17.08%

## Exercise 4.5
A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the `__syncthreads()` instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

**Ans**: Bad idea. We cannot assume that the warp size is always 32.

## Exercise 4.6
If a CUDA device’s SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?
a. 128 threads per block
b. 256 threads per block 
c. 512 threads per block 
d. 1024 threads per block

**Ans**: We should pick the total number of threads that is larger than the number of threads taken in SM. 512 * 4 = 2048 threads is a good choice for the latency tolerance.

## Exercise 4.7
Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level.
a. 8 blocks with 128 threads each
b. 16 blocks with 64 threads each
c. 32 blocks with 32 threads each
d. 64 blocks with 32 threads each 
e. 32 blocks with 64 threads each

**Ans**: 
a. 8 blocks with 128 threads each
- Number of threads: 8*128 = 1024 threads
- Occupancy: 1024/2048 = 50%
b. 16 blocks with 64 threads each
- Number of threads: 16*64 = 1024 threads
- Occupancy: 1024/2048 = 50%
c. 32 blocks with 32 threads each
- Number of threads: 32*32 = 1024 threads
- Occupancy: 1024/2048 = 50%
d. 64 blocks with 32 threads each
- Number of threads: 64*32 = 2048 threads
- Occupancy: 2048/2048 = 100%
e. 32 blocks with 64 threads each
- Number of threads: 32*64 = 2048 threads
- Occupancy: 2048/2048 = 100%

## Exercise 4.8
Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.
a. The kernel uses 128 threads per block and 30 registers per thread. 
b. The kernel uses 32 threads per block and 29 registers per thread.
c. The kernel uses 256 threads per block and 34 registers per thread.

**Ans**: 
a. 128 threads per block and 30 registers per thread
-  Maximum number of blocks this kernel can have:
    2048/128 = 16 blocks
- Total number of registers will be used : 16 x 128 x 30 = 61440 registers
- This is below the limit of 64K registers per SM.
- Thus, this kernel can achieve full occupancy as all 2048 threads can be accommodated in the SM.

b. 32 threads per block and 29 registers per thread
- Maximum number of blocks this kernel can have:
    2048/32 = 64 blocks
    But only 32 blocks can be accommodated in the SM.
- Total number of registers will be used : 32 x 32 x 29 = 29440 registers
- This is below the limit of 64K registers per SM.
- Kernel can only launch 32 x 32 = 1024 threads in the SM. Thus, the kernel can only achieve 50% occupancy.
- The limiting factor is the number of blocks that can be accommodated in the SM.

c. 256 threads per block and 34 registers per thread
- Maximum number of blocks this kernel can have:
    2048/256 = 8 blocks
    This is below the limit of 32 blocks per SM.
- Total number of registers will be used : 8 x 256 x 34 = 69632 registers
- This is above the limit of 64K registers per SM and will be a limiting factor.
- Total number of blocks that can be launched = 64000/ (256 x 34) = 7 blocks
- Kernel can only launch 7 x 256 = 1792 threads in the SM. Thus, the kernel can only achieve 87.5% occupancy.

## Exercise 4.9
A student mentions that they were able to multiply two 1024x1024 matrices using a matrix multiplication kernel with 32 x 32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

**Ans**: I would question the correctness of their claim because their choice of thread block size (32×32 = 1024 threads per block) exceeds the CUDA device's limit of 512 threads per block. This means their kernel should not work as described on their given hardware.
