
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

#define BATCH 256
#define INPUT_SIZE 128
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define REPEATS 20

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error at " << __FILE__ << ":"      \
                  << __LINE__ << " - "                       \
                  << cudaGetErrorString(err) << std::endl;   \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

// Matrix multiplication with bias
__global__ void matrixBiasKernel(
    const float* A, const float* B, const float* bias,
    float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = bias[col];

        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];

        C[row * N + col] = sum;
    }
}

// ReLU activation
__global__ void reluKernel(float* data, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        data[i] = fmaxf(0.0f, data[i]);
}

// Sigmoid activation
__global__ void sigmoidKernel(float* data, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        data[i] = 1.0f / (1.0f + expf(-data[i]));
}

// CPU forward pass
void cpuForwardPass(
    const std::vector<float>& X,
    const std::vector<float>& W1,
    const std::vector<float>& b1,
    const std::vector<float>& W2,
    const std::vector<float>& b2,
    std::vector<float>& output)
{
    std::vector<float> hidden(BATCH * HIDDEN_SIZE);

    for (int i = 0; i < BATCH; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            float sum = b1[j];

            for (int k = 0; k < INPUT_SIZE; ++k)
                sum += X[i * INPUT_SIZE + k] *
                       W1[k * HIDDEN_SIZE + j];

            hidden[i * HIDDEN_SIZE + j] = std::max(0.0f, sum);
        }
    }

    for (int i = 0; i < BATCH; ++i) {
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            float sum = b2[j];

            for (int k = 0; k < HIDDEN_SIZE; ++k)
                sum += hidden[i * HIDDEN_SIZE + k] *
                       W2[k * OUTPUT_SIZE + j];

            output[i * OUTPUT_SIZE + j] =
                1.0f / (1.0f + std::exp(-sum));
        }
    }
}

// GPU forward pass
void launchGpuForwardPass(
    float* dX, float* dW1, float* db1, float* dZ1,
    float* dW2, float* db2, float* dZ2)
{
    dim3 block(16, 16);
    dim3 grid1(
        (HIDDEN_SIZE + 15) / 16,
        (BATCH + 15) / 16
    );

    matrixBiasKernel<<<grid1, block>>>(
        dX, dW1, db1, dZ1,
        BATCH, HIDDEN_SIZE, INPUT_SIZE
    );

    int hiddenCount = BATCH * HIDDEN_SIZE;

    reluKernel<<<(hiddenCount + 255) / 256, 256>>>(
        dZ1, hiddenCount
    );

    dim3 grid2(
        (OUTPUT_SIZE + 15) / 16,
        (BATCH + 15) / 16
    );

    matrixBiasKernel<<<grid2, block>>>(
        dZ1, dW2, db2, dZ2,
        BATCH, OUTPUT_SIZE, HIDDEN_SIZE
    );

    int outputCount = BATCH * OUTPUT_SIZE;

    sigmoidKernel<<<(outputCount + 255) / 256, 256>>>(
        dZ2, outputCount
    );

    CUDA_CHECK(cudaGetLastError());
}

// Test matrix multiplication with bias
bool testMatrixKernel()
{
    const int M = 2, N = 2, K = 2;

    float hA[] = {1, 2, 3, 4};
    float hB[] = {1, 2, 3, 4};
    float hBias[] = {1, 1};
    float hC[4] = {};

    float *dA, *dB, *dBias, *dC;

    CUDA_CHECK(cudaMalloc((void**)&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc((void**)&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc((void**)&dBias, sizeof(hBias)));
    CUDA_CHECK(cudaMalloc((void**)&dC, sizeof(hC)));

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBias, hBias, sizeof(hBias), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(1, 1);

    matrixBiasKernel<<<grid, block>>>(
        dA, dB, dBias, dC, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));

    bool passed =
        std::fabs(hC[0] - 8.0f) < 1e-5f &&
        std::fabs(hC[1] - 11.0f) < 1e-5f &&
        std::fabs(hC[2] - 16.0f) < 1e-5f &&
        std::fabs(hC[3] - 23.0f) < 1e-5f;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dBias);
    cudaFree(dC);

    return passed;
}

// Test ReLU
bool testReLUKernel()
{
    float input[] = {-2, -1, 0, 1, 2};
    float output[5];

    float* dData;

    CUDA_CHECK(cudaMalloc((void**)&dData, sizeof(input)));
    CUDA_CHECK(cudaMemcpy(dData, input, sizeof(input),
                          cudaMemcpyHostToDevice));

    reluKernel<<<1, 256>>>(dData, 5);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(output, dData, sizeof(output),
                          cudaMemcpyDeviceToHost));

    bool passed =
        output[0] == 0 &&
        output[1] == 0 &&
        output[2] == 0 &&
        output[3] == 1 &&
        output[4] == 2;

    cudaFree(dData);
    return passed;
}

// Test Sigmoid
bool testSigmoidKernel()
{
    float input[] = {-2, -1, 0, 1, 2};
    float output[5];

    float* dData;

    CUDA_CHECK(cudaMalloc((void**)&dData, sizeof(input)));
    CUDA_CHECK(cudaMemcpy(dData, input, sizeof(input),
                          cudaMemcpyHostToDevice));

    sigmoidKernel<<<1, 256>>>(dData, 5);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(output, dData, sizeof(output),
                          cudaMemcpyDeviceToHost));

    bool passed =
        std::fabs(output[0] - 0.119203f) < 1e-4f &&
        std::fabs(output[1] - 0.268941f) < 1e-4f &&
        std::fabs(output[2] - 0.5f) < 1e-4f &&
        std::fabs(output[3] - 0.731059f) < 1e-4f &&
        std::fabs(output[4] - 0.880797f) < 1e-4f;

    cudaFree(dData);
    return passed;
}

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable NVIDIA GPU detected.\n";
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "EXPERIMENT 5: NEURAL NETWORK LAYERS IN CUDA\n";
    std::cout << "GPU detected: " << prop.name << "\n\n";

    std::cout << "Testing individual CUDA kernels...\n";

    bool matrixOK = testMatrixKernel();
    bool reluOK = testReLUKernel();
    bool sigmoidOK = testSigmoidKernel();

    std::cout << "Matrix multiplication + bias: "
              << (matrixOK ? "PASSED" : "FAILED") << "\n";

    std::cout << "ReLU: "
              << (reluOK ? "PASSED" : "FAILED") << "\n";

    std::cout << "Sigmoid: "
              << (sigmoidOK ? "PASSED" : "FAILED") << "\n";

    if (!(matrixOK && reluOK && sigmoidOK))
        return EXIT_FAILURE;

    std::vector<float> X(BATCH * INPUT_SIZE);
    std::vector<float> W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> b1(HIDDEN_SIZE);
    std::vector<float> W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> b2(OUTPUT_SIZE);

    for (size_t i = 0; i < X.size(); ++i)
        X[i] = 0.5f * std::sin(static_cast<float>(i) * 0.013f);

    for (size_t i = 0; i < W1.size(); ++i)
        W1[i] = 0.08f * std::cos(static_cast<float>(i) * 0.017f);

    for (size_t i = 0; i < b1.size(); ++i)
        b1[i] = 0.01f * static_cast<float>(i % 7);

    for (size_t i = 0; i < W2.size(); ++i)
        W2[i] = 0.10f * std::sin(static_cast<float>(i) * 0.019f);

    for (size_t i = 0; i < b2.size(); ++i)
        b2[i] = 0.02f * static_cast<float>(i % 5);

    std::vector<float> cpuOutput(BATCH * OUTPUT_SIZE);

    auto cpuStart = std::chrono::steady_clock::now();

    for (int r = 0; r < REPEATS; ++r)
        cpuForwardPass(X, W1, b1, W2, b2, cpuOutput);

    auto cpuStop = std::chrono::steady_clock::now();

    double cpuMs =
        std::chrono::duration<double, std::milli>(
            cpuStop - cpuStart).count() / REPEATS;

    float *dX, *dW1, *db1, *dZ1, *dW2, *db2, *dZ2;

    CUDA_CHECK(cudaMalloc((void**)&dX, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dW1, W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&db1, b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dZ1, BATCH * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dW2, W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&db2, b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dZ2, BATCH * OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, W1.data(), W1.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db1, b1.data(), b1.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, W2.data(), W2.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db2, b2.data(), b2.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t eventStart, eventStop;

    CUDA_CHECK(cudaEventCreate(&eventStart));
    CUDA_CHECK(cudaEventCreate(&eventStop));

    CUDA_CHECK(cudaEventRecord(eventStart));

    for (int r = 0; r < REPEATS; ++r)
        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);

    CUDA_CHECK(cudaEventRecord(eventStop));
    CUDA_CHECK(cudaEventSynchronize(eventStop));

    float gpuKernelMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &gpuKernelMs, eventStart, eventStop));

    gpuKernelMs /= REPEATS;

    std::vector<float> gpuOutput(BATCH * OUTPUT_SIZE);

    auto gpuTotalStart = std::chrono::steady_clock::now();

    for (int r = 0; r < REPEATS; ++r) {

        CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(dW1, W1.data(), W1.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(db1, b1.data(), b1.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(dW2, W2.data(), W2.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(db2, b2.data(), b2.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        launchGpuForwardPass(dX, dW1, db1, dZ1, dW2, db2, dZ2);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            gpuOutput.data(), dZ2,
            gpuOutput.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    auto gpuTotalStop = std::chrono::steady_clock::now();

    double gpuEndToEndMs =
        std::chrono::duration<double, std::milli>(
            gpuTotalStop - gpuTotalStart).count() / REPEATS;

    float maxDifference = 0.0f;

    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        maxDifference = std::max(
            maxDifference,
            std::fabs(cpuOutput[i] - gpuOutput[i]));
    }

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\nForward-pass dimensions:\n";
    std::cout << "Input: " << BATCH << " x " << INPUT_SIZE << "\n";
    std::cout << "Hidden layer: " << BATCH << " x " << HIDDEN_SIZE << "\n";
    std::cout << "Output: " << BATCH << " x " << OUTPUT_SIZE << "\n";

    std::cout << "\nAverage CPU compute time: "
              << cpuMs << " ms\n";

    std::cout << "Average GPU kernel-only time: "
              << gpuKernelMs << " ms\n";

    std::cout << "Average GPU end-to-end time: "
              << gpuEndToEndMs << " ms\n";

    if (gpuKernelMs > 0.0f)
        std::cout << "Kernel-only speedup: "
                  << cpuMs / gpuKernelMs << "x\n";

    if (gpuEndToEndMs > 0.0)
        std::cout << "End-to-end speedup: "
                  << cpuMs / gpuEndToEndMs << "x\n";

    std::cout << "\nMaximum absolute output difference: "
              << maxDifference << "\n";

    std::cout << "CPU and GPU outputs: "
              << (maxDifference < 1e-4f
                  ? "MATCH (within tolerance)"
                  : "DO NOT MATCH") << "\n";

    std::cout << "\nFirst 10 output values (GPU):\n";

    for (int i = 0; i < 10; ++i)
        std::cout << gpuOutput[i]
                  << (i == 9 ? '\n' : ' ');

    CUDA_CHECK(cudaEventDestroy(eventStart));
    CUDA_CHECK(cudaEventDestroy(eventStop));

    cudaFree(dX);
    cudaFree(dW1);
    cudaFree(db1);
    cudaFree(dZ1);
    cudaFree(dW2);
    cudaFree(db2);
    cudaFree(dZ2);

    return (maxDifference < 1e-4f)
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
