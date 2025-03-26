# Stencil
- Stencil sweeps are done on **three-dimensional grids**
- Convolution is done on **two-dimensional images**/a small number of time slices of 2D images
---
- Stencil patterns may enable **register** tiling of input data
- Convolution may enable **shared memory** tiling of input data

## Background
Stencils are foundational to numerical methods for solving partial differential equations

Stencils are used to calculate an approximate derivative value (output) at each grid point from function values at the neighboring grid points and the grid point itself.

## Important
|**CUDA Concept**|**Limit**|
|:--------------:|:-------:|
|Threads per block|Max:1024|
|Grid size (x-dim)|2^31 - 1|
|Grid size (y-dim, z-dim)|65536|

## Summary
Two major disadvantages of the small tile size:

1. Limit the reuse ratio and thus the compute to memory access ratio.
- For example, for a 3D stencil of order 1, an 8×8×8 3D input tile has 512 elements. The corresponding output tile has 6×6×6=216 elements, which means that 512−216=296 of the input elements are halo elements. The portion of **halo elements** in the input tile is about **58%**.
2. Has an adverse impact on memory coalescing. These accesses cannot be coalesced and will **underutilize the DRAM bandwidth**. 
