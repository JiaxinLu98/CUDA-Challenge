# Compute architecture and scheduling

I learned some basic concepts today, but they provided me with a solid understanding of parallel programming, what to consider in the future to optimize code from both an algorithmic and memory perspective, and the reasons behind these considerations.

## Table of Contents
1. [Architecture of a modern GPU](#1-architecture-of-a-modern-gpu)
2. [Block scheduling](#2-block-scheduling)
3. [Synchronization and transparent scalability](#3-synchronization-and-transparent-scalability)
4. [Warps and SIMD hardware](#4-warps-and-simd-hardware)
5. 

## 1. Architecture of a modern GPU

### Terms
- *Streaming multiprocessors (SMs)*: Each SM has several streaming processors / CUDA cores that share control logic and memory resources.
- *Global Memory*: DRAM

## 2. Block scheduling

### Blocks Need to Reserve Hardware Resources

Only a **limited** number of blocks can be simultaneously assigned to a given SM. The assignment of threads to SMs on a block-by-block basis guarantees that threads in the same block are scheduled simultaneously on the same SM. 

## 3. Synchronization and transparent scalability

### Barrier Synchronization

CUDA allows **threads in the same block** to coordinate their activities using the barrier synchronization function `__syncthreads()`. When a thread calls `__syncthreads()`, it will be held at the program location of the call until every thread in the same block reaches that location. 

---

### `__syncthreads()` in if statement / if-then-else statement

In an **if** statement, either **all** threads in a block execute the path that includes the `__syncthreads()` or **none** of them does.

In an **if-then-else** statement, not all threads in a block are guaranteed to execute either of the barriers so it will result in an incorrect result or a *deadlock*.

Assigning execution resources to **all threads in a block as a unit** ensures the time proximity of all threads in a block and prevents an excessive or even indefinite waiting time during barrier synchronization.

---

### Scalability

By not allowing threads in different blocks to perform barrier synchronization with each other, the CUDA runtime system can execute blocks **in any order** relative to each other, since none of them need to wait for each other. This flexibility enables scalable implementations

*Transparent Scalability*: The ability to execute the **same application code** on **different hardware** with **different amounts of execution resources**.

## 4. Warps and SIMD hardware

### Thread Scheduling

*Warp*: Once a block has been assigned to an SM, it is further divided into 32-thread units. It is the **unit** of thread scheduling in SMs.

For example, if each block has 256 threads, we can determine that each block has 256/32 or 8 warps. With three blocks in the SM, we have 3*8=24 warps in the SM.

Block is a **one-dimensional** array: warp n starts with thread **32xn** and ends with thread **32x(n+1)-1**. For a block whose size is not a multiple of 32, the last warp will be padded with **inactive threads** to fill up the 32 thread positions.

Block consists of multiple dimensions of threads, the dimensions will be projected into a **linearized row-major** layout before partitioning into warps.

---

### SIMD Model

Threads in the same warp are assigned to the same processing block, which fetched the instruction for the warp and executes it for all threads in the warp at the same time. Those threads apply the same instruction to different portions of the data.

## 
