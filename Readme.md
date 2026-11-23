
# PGC Lab Experiment 5
## Implementation of Neural Network Layers Using CUDA

### 1. Aim
To implement basic neural network layers using CUDA C++ and compare CPU and GPU execution performance.

### 2. Objectives
- Implement matrix multiplication with bias using CUDA.
- Implement ReLU activation function.
- Implement Sigmoid activation function.
- Perform neural network forward propagation.
- Compare CPU and GPU execution times.
- Analyze GPU acceleration and output correctness.

### 3. Hardware and Software Requirements

**Hardware used for the original experiment:**
- GPU: NVIDIA RTX 4500 Ada Generation
- GPU Memory: 24 GB
- CPU: College PC processor

**Software:**
- NVIDIA CUDA Toolkit 13.4
- Microsoft Visual Studio 2026
- CUDA C++
- Windows 11

### 4. Neural Network Architecture

| Layer | Dimensions |
|---|---|
| Input Layer | 256 × 128 |
| Hidden Layer | 256 × 64 |
| Output Layer | 256 × 10 |

### 5. CUDA Kernels Implemented

**1. Matrix Multiplication with Bias**

Computes matrix multiplication and adds the corresponding bias values.

**2. ReLU Activation**

ReLU is defined as:

f(x) = max(0, x)

**3. Sigmoid Activation**

Sigmoid is defined as:

f(x) = 1 / (1 + e^(-x))

### 6. Compilation

The original experiment was compiled on the college PC using:

```cmd
nvcc -O2 -arch=compute_89 -code=sm_89 neural_network.cu -o neural_network.exe
```

### 7. Execution

```cmd
neural_network.exe
```

### 8. Experimental Results

Original measurements recorded on the college PC:

| Parameter | Observed Value |
|---|---:|
| CPU compute time | 0.913715 ms |
| GPU kernel-only time | 0.021510 ms |
| GPU end-to-end time | 0.064525 ms |
| Kernel-only speedup | 42.477825× |
| End-to-end speedup | 14.160635× |
| Maximum output difference | 0.000000 |

**Kernel Validation:**
- Matrix Multiplication + Bias: PASSED
- ReLU: PASSED
- Sigmoid: PASSED
- CPU/GPU Output: MATCH

### 9. Performance Analysis

The GPU implementation achieved approximately 42.48× kernel-only speedup and 14.16× end-to-end speedup compared with the CPU implementation in the original college-PC experiment.

The CPU and GPU outputs matched within the specified tolerance.

### 10. Conclusion

The experiment demonstrated the implementation of basic neural network layers using CUDA C++. Matrix multiplication with bias, ReLU, and Sigmoid activation functions were successfully implemented and tested.

The performance comparison demonstrated the benefits of GPU acceleration for neural network computations.

---

**Course:** Parallel and GPU Computing (PGC)  
**Experiment:** 5  
**Language:** CUDA C++
