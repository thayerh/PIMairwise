# FourierPIM: High-Throughput In-Memory Fast Fourier Transform and Polynomial Multiplication
## Overview
This is the simulation environment for the following paper, 

`Orian Leitersdorf, Yahav Boneh, Gonen Gazit, Ronny Ronen, and Shahar Kvatinsky, “FourierPIM: High-Throughput In-Memory Fast Fourier Transform and Polynomial Multiplication,” Memories - Materials, Devices, Circuits and Systems, 2023.` 

The goal of the simulator is to verify the correctness of the algorithms proposed in the paper,
to measure the performance of the algorithms (latency, area, energy), and to serve as an open-source library that 
may be utilized in other works. The simulator consists of three separate parts: a logical simulator for the functionality
of a memristor crossbar array, a set of implementations for the algorithms proposed in FourierPIM, and a set of testers that  
verify the correctness of the proposed algorithms and also measure performance. Below we include details on the execution of the
simulator, as well as more detailed explanations for the three parts of the simulator.

See the `gpu` folder for the derivation of the baseline GPU results.

## User Information
### Dependencies
The simulation environment is implemented via `numpy` to enable fast bitwise operations. Therefore, 
the project requires the following libraries:
1. python3
2. numpy

### Organization

1. `simulator`: Includes the logical simulation for a memristive crossbar array.
2. `algorithms`: Includes all the algorithms developed for FourierPIM, alongside an adaption of the AritPIM [1] algorithms
for floating-point arithmetic. Recent additions include:
   - `pwxc.py`: Pairwise cross-correlation algorithm
   - `pegasos.py`: SVM training using the Pegasos algorithm
   - `bbf.py`: Butterworth bandpass filter implementation
3. `test`: Includes the testers for the proposed algorithms, with comprehensive test suites for all implemented algorithms.
4. `util`: Miscellaneous helper functions (e.g., converting between different number representations, inverse square root, Pegasos utilities).
5. `impl`: Contains alternative implementations and optimizations including C, LLVM IR, and CUDA versions of key algorithms for performance comparison.
6. `gpu`: GPU baseline implementations for benchmarking PIM performance against traditional GPU architectures.

### Logical Simulation

The fundamental idea of the logical simulator is to represent a memristor crossbar array as a binary matrix that supports
bitwise operations on rows and columns of the array. Specifically, we assume the following:

1. The memory supports the NOT/MIN3 set of logic gates, proposed by FELIX [2]. 
2. Only a single initialization is allowed per cycle. When initialization cycles are not performed, then the result is ANDed with the previous value of the cell (see FELIX [2]).
3. The memory supports a write operation that writes a single number to a single location in each crossbar in a single cycle.

### Proposed Algorithms

The proposed algorithms are divided into six parts:

#### Proposed Algorithms: Arithmetic

We adapt the arithmetic functions from AritPIM [1] within the simulation code. Further, we extend the 
functions to support arithmetic with complex numbers. Overall, we assume only half-precision and full-precision numbers
for simplicity, corresponding to the IEEE 16-bit and IEEE 32-bit standards.

**New Arithmetic Methods Added:**
- **`dot()`**: Performs dot product operations on vectors using tree-based reduction with element-wise multiplication and iterative addition
- **`batch_dot()`**: Batched version of dot product for efficient processing of multiple vector pairs simultaneously
- **`unzip()`**: Data unpacking operation for reorganizing memory layouts
- **`approxReciprocalSqrt()`**: Fast approximation of 1/√x using the "magic number" trick (Quake III algorithm) for applications where rough estimates are sufficient
- **`__and()`**: Bitwise AND operation implemented using NOT and MIN3 gates
- **Extended DataType**: Added bias field to IEEE float representations for improved numerical handling

#### Proposed Algorithms: FFT

We begin by extending the complex arithmetic towards an element-parallel butterfly operation. Then, we provide
implementations for each of the configurations (r, 2r, 2rbeta) by utilizing both the butterfly operations and a sequence of swap operations. The swap operations are performed without requiring additional intermediate rows by utilizing
the intermediate columns that are typically used for arithmetic.

#### Proposed Algorithms: Polynomial Multiplication

We extend the FFT towards polynomial multiplication through the convolution theorem. While complex polynomial multiplication
follows immediately from the proposed FFT, we find that real FFT requires several techniques to perform the FFT packing efficiently
(i.e., computing the reverse conjugate and applying it to the correct locations requires complex swap operations).

#### Proposed Algorithms: Pairwise Cross-Correlation (PWXC)

We implement pairwise cross-correlation algorithms that compute correlations between all pairs of input signals. The implementation 
uses element-wise multiplication followed by tree-based reduction to efficiently sum across rows. This algorithm is particularly 
useful for signal processing applications and follows the approach described in the FORESEE paper.

#### Proposed Algorithms: SVM Training (Pegasos)

We implement the Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm for training Support Vector Machines in PIM. 
The implementation includes both standard and batched forms of the algorithm. This extends the PIM arithmetic operations to 
support machine learning workloads, including dot products, vector operations, and iterative gradient descent updates. The algorithm
includes implementations in C, LLVM IR, and CUDA for performance comparison.

#### Proposed Algorithms: Butterworth Bandpass Filter (BBF)

We implement a Butterworth bandpass filter for signal processing applications in PIM. The BBF implementation computes filter 
coefficients and applies IIR filtering to signals stored in PIM arrays. This algorithm demonstrates the capability of PIM 
architectures to handle digital signal processing workloads including frequency filtering and power computation.


## References
[1] O. Leitersdorf, D. Leitersdorf, J. Gal, M. Dahan, R. Ronen, and S. Kvatinsky, "AritPIM: High-Throughput In-Memory Arithmetic," 2022.

[2] S. Gupta, M. Imani and T. Rosing, "FELIX: Fast and Energy-Efficient Logic in Memory," IEEE/ACM International Conference on Computer-Aided Design (ICCAD), 2018, pp. 1-7.

## Recent Updates

### December 2025
- **BBF Implementation**: Added Butterworth bandpass filter (BBF) for digital signal processing applications including frequency filtering and power computation.
- **SVM Enhancements**: Extended Pegasos SVM implementation with batch processing support and additional implementations in C, LLVM IR, and CUDA for performance comparison.
- **PWXC Updates**: Updated pairwise cross-correlation algorithm to reflect the approach described in the FORESEE paper, with improved efficiency and correctness.
- **SVM Training**: Implemented the Pegasos algorithm for Support Vector Machine training in both standard and batched forms, extending PIM capabilities to machine learning workloads.
- **Arithmetic Extensions**: Enhanced arithmetic.py with new operations to support machine learning and advanced numerical computations:
  - Dot product operations (standard and batched) for vector computations
  - Fast inverse square root approximation using bit manipulation techniques
  - Bitwise AND operation and data unpacking utilities
  - Improved IEEE float handling with bias field support
- **GPU Improvements**: Fixed GPU baseline bugs and updated GPU benchmarking code for more accurate performance comparisons.
- **Test Coverage**: Added comprehensive test suites for BBF, SVM, PWXC, and arithmetic operations.
