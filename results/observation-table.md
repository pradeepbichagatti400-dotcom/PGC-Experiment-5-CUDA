
# Experiment 5 - Observation Table

## Hardware
- GPU: NVIDIA RTX 4500 Ada Generation
- CUDA Toolkit: 13.4
- GPU Architecture: Ada Lovelace (SM 8.9)

## Neural Network Configuration

| Layer | Dimensions |
|---|---|
| Input | 256 × 128 |
| Hidden | 256 × 64 |
| Output | 256 × 10 |

## Performance Measurements

Original measurements recorded on the college PC.

| Measurement | Observed Value |
|---|---:|
| CPU compute time | 0.913715 ms |
| GPU kernel-only time | 0.021510 ms |
| GPU end-to-end time | 0.064525 ms |
| Kernel-only speedup | 42.477825× |
| End-to-end speedup | 14.160635× |
| Maximum output difference | 0.000000 |

## Kernel Validation

| CUDA Kernel | Result |
|---|---|
| Matrix Multiplication + Bias | PASSED |
| ReLU Activation | PASSED |
| Sigmoid Activation | PASSED |
| CPU/GPU Output Comparison | MATCH |

## Conclusion

The experiment demonstrated CUDA-based neural network layer implementation and GPU acceleration. All three individual CUDA kernel tests passed, and CPU and GPU outputs matched within tolerance on the original college PC.
