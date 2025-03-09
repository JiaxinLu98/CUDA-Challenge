# Introduction to Parallel Programming

I learned some basic concepts today, but they provided me with a solid understanding of parallel programming, what to consider in the future to optimize code from both an algorithmic and memory perspective, and the reasons behind these considerations.

## Table of Contents
1. [Heterogeneous parallel computing](#1-heterogeneous-parallel-computing)
2. [Why more speed or parallelism?](#2-why-more-speed-or-parallelism)
3. [Speeding up real applications](#3-speeding-up-real-applications)

## 1. Heterogeneous parallel computing

### Two main paths for designing microprocessors
- *Multicore* refers to a processor that has more than one logical CPU core, and that can physically execute multiple instructions at the same time. (MIMD)
- *Multithreading* refers to a program that can take advantage of a multicore computer by runnning on more than one core at the same time. (SIMD / MIMD)

---

### CPU and GPU are designed differently

The difference makes a large peak performance gap between many-threaded GPUs and multicore CPUs.

The table illustrates the peak floating-point throughput
| **Type of Processor** | **64-bit double-precision** | **32-bit single-precision** | **16-bit half-precision** |
|:---------------------:|:---------------------------:|:---------------------------:|:-------------------------:|
|      **A100 GPU**     |        **9.7 TFLOPS**       |        **156 TFLOPS**       |        **312 TFLOPS**     |
|    **24-core CPU**    |        **0.33 TFLOPS**      |        **0.66 TFLOPS**      |                           |

- *CPU (latency-oriented design)* reduces execution latency per thread but at the cost of chip area and power.
    - High-speed Arithmetic Units --> Designed for low latency computations
    - Large Last-Level Caches --> Reduce memory access latency by storing frequently used data closer to the CPU
    - Branch Prediction & Execution Control --> Mitigate latency of conditional branches by predicting and pre-executing instructions
- *GPU (throughput-oriented design)* prioritizes many parallel execution units, sacrificing latency per thread for overall throughput.

## 2. Why more speed or parallelism?

When an application is suitable for parallel execution, a good implementation on a GPU can achieve a speed up of more than 100 times over sequential execution on a single CPU core. If the application includes what we call “data parallelism,” it is often possible to achieve a 10× speedup with just a few hours of work.

## 3. Speeding up real applications

### Speedup

$$
speedup = \frac{T_B}{T_A}
$$

This equation shows how much faster system A is compared to system B.

---

### First factor: Parallel portion 

**Amdahl's Law** states that the maximum speed of a task is related to the sum of **parallel** and **serial** components. It is very important that an application has the vast majority of its execution in the parallel portion for a massively parallel processor to effectively speed up its execution.

Therefore, extensive optimization and tuning for **algorithms** are needed.

---

### Second factor: How fast data can be accessed from and written to the memory

Straightforward parallelization often saturates memory (DRAM) bandwidth.

To figure out how to get around **memory bandwith limitations**, we need to utilize specialized **GPU on-chip memories** to reduce the number of accesses to the DRAM.

Therefore, extensive optimization and tuning for **limited on-chip memory** are needed.
