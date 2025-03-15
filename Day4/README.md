# Memory Architecture and Data Locality

## Table of Contents
1. [Importance of memory access efficiency](#1-importance-of-memory-access-efficiency)
2. [CUDA memory types](#2-cuda-memory-types)
3. 

## 1. Importance of memory access efficiency

### Effect of Memory Access Efficiency
Calculate the expected performance level of **the most executed** portion of the kernel code.

*Compute to global memory access ratio/arithmetic intensity/ computational intensity*: the ratio of floating-point operations (FLOP) to bytes (B) accessed from global memory --> the number of FLOPs performed for each byte access from the global memory within a region of a program.   

---

### Impact on CUDA Kernel Performance
For example, the A100 GPU's peak global memory bandwidth is 1555 GB/second. The matrix multiplication kernel performs 0.25 OP/B, throughput: `155GB/second * 0.25OP/B = 389giga FLOPs/second (GFLOPS)` which is limited by the global memory bandwidth. The throughput of the A100 is 19,500 GFLOPS, 389 GFLOPS is only 2% of the peak throughput of the A100. 

*Memory bandwidth programs*: programs whose execution speed is limited by memory bandwidth.

## 2. CUDA memory types
|  **Memory Type**  |  **Host**  |  **Device**  |  **Shared across threads**  |
|:-----------------:|:----------:|:------------:|:---------------------------:|
| **Global Memory** | **Read & Write** | **Read & Write** | **Yes** |
| **Constant Memory** | **Read & Write** | **Short-Latency, High-Bandwidth Read-Only** | **Yes** |
| **Local Memory (in the global memory)** | **Read & Write** | **Read & Write** | **No** |
| **Shared Memory**| **No** | **Yes** | **Yes**|
| **Register**| **No** | **Yes** | **No**|

### Reason why placing the operands in registers
- Each access to registers involves **fewer instructions** than an access to the global memory.
- Since the **processor** can fetch and execute only **a limited number of instructions per clock cycle**, the version with an additional load will likely **take more time** to process than the one without.
- In modern computers the **energy** that is consumed for accessing a value from the register file is at least an order of magnitude lower than for accessing a value from the global memory. The **occupancy** that is achieved for an application can be **reduced** if the register usage in **full-occupancy** scenarios **exceeds the limit**. Therefore we also need to avoid oversubscribing to this limited resource whenever possible.

---

### Differences between the shared memory and registers
When the processor accesses data that resides in the shared memory, it needs to **perform a memory load operation**. Shared memory can be accessed with much lower latency and much higher throughput than the global memory. Because of the need to perform a load operation, shared memory has longer latency and lower bandwidth than registers.

**Variables that reside in the shared memory are accessible by all threads in a block**.  

---

### CUDA variable declaration type
- *__shared__*: Within a thread block; A private version of the shared variable is created for and used by each block during kernel execution; The lifetime of a shared variable is within the duration of the kernel execution.
- *__constant__*: Outside any function body; The scope of a constant variable is all grids, meaning that all threads in all grids see the same version of a constant variable; The lifetime of a constant variable is the entire application execution; Constant variables are often used for variables that provide input values to kernel functions. The values of the constant variables cannot be changed by the kernel function code.
- *__device__*: Within the global memory; Global variables are often used to pass information from one kernel invocation to another kernel invocation.

