#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <fstream>
#include <chrono>  // For timing
#include <random>  // For weight initialization
#include <cmath>   // For math operations like fabs
#include <sstream>
#include <string>
#include <algorithm> // Add this at the beginning of your file
#include <curand.h>

/*
// Define a reduction kernel for summing the errors
__global__ void reduceBiasGradientKernel(float* input, float* output, int size, int batchSize) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Kernel Debug: Thread %d, Block %d\n", tid, blockIdx.x);

    // Initialize shared memory with input values
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;

    __syncthreads();

    // Perform reduction within a block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Store the block sum in the output
    if (tid == 0) {
        output[blockIdx.x] = sdata[0] / static_cast<float>(batchSize);
    }
}
*/

// Function to save weights to a text file
void saveWeights(const std::vector<std::vector<float>>& weights, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing weights: " << filename << std::endl;
        return;
    }

    // Iterate through each layer of weights
    for (const auto& layerWeights : weights) {
        // Save each weight in the layer
        for (float weight : layerWeights) {
            file << weight << " ";
        }
        // New line for a new layer
        file << std::endl;
    }
    file.close();
}

// Function to save biases to a text file
void saveBiases(const std::vector<std::vector<float>>& biases, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing biases: " << filename << std::endl;
        return;
    }

    // Iterate through each layer of biases
    for (const auto& layerBiases : biases) {
        // Save each bias in the layer
        for (float bias : layerBiases) {
            file << bias << " ";
        }
        // New line for a new layer
        file << std::endl;
    }
    file.close();
}

__global__ void applyDropout(float* layerOutput, float* dropoutMask, float dropoutRate, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        bool drop = dropoutMask[idx] < dropoutRate;
        layerOutput[idx] = drop ? 0.0f : layerOutput[idx] / (1.0f - dropoutRate); // Optionally scale the activations
    }
}

#define CURAND_CHECK_ERROR(status) { \
    if (status != CURAND_STATUS_SUCCESS) { \
        std::cerr << "CURAND API failed at line " << __LINE__ << " with error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}


// Function to transpose a matrix on the CPU
void transposeMatrixCPU(const std::vector<float>& input, std::vector<float>& output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j]; // Transpose operation
        }
    }
}


void printDeviceArray(float* d_array, int size, const std::string& arrayName) {
    std::vector<float> h_array(size);
    cudaMemcpy(h_array.data(), d_array, size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cerr << arrayName << ": ";
    for (int i = 0; i < size; ++i) {
        std::cerr << h_array[i] << " ";
    }
    std::cerr << std::endl;
}

bool areLossesIncreasing(const std::vector<float>& losses, int windowSize) {
    if (losses.size() < windowSize) return false;

    for (int i = losses.size() - windowSize; i < losses.size() - 1; ++i) {
        if (losses[i] <= losses[i + 1]) {
            return false;
        }
    }

    return true;
}


float replaceNaN(float value) {
    if (std::isnan(value)) {
        // Replace NaN with a suitable value, for example, 0.0
        return 0.0;
    }
    return value;
}

__global__ void applyGradientClipping(float* gradients, int size, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Clip the gradient if it exceeds the threshold
        if (gradients[idx] > threshold) {
            gradients[idx] = threshold;
        } else if (gradients[idx] < -threshold) {
            gradients[idx] = -threshold;
        }
    }
}


__global__ void reduceBiasGradientKernel(float* input, float* output, int size, int batchSize) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0] / static_cast<float>(batchSize);
        printf("Block %d reduced sum: %f\n", blockIdx.x, sdata[0]);

        // Debug logs to check input data
        printf("Block %d input data: ", blockIdx.x);
        for (int i = 0; i < size; ++i) {
            printf("%f ", input[i]);
        }
        printf("\n");
    }
}

__global__ void applyReluDerivative(float* d_previousLayerError, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        if (d_previousLayerError[idx] > 0.0f) {
            d_previousLayerError[idx] = 1.0f;
        } else {
            d_previousLayerError[idx] = 0.0f;
        }
    }
}

__global__ void applyReluDerivative2(float* layerErrors, const float* layerOutputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        layerErrors[idx] *= layerOutputs[idx] > 0.0f ? 1.0f : 0.0f;
    }
}



// CUDA Kernel for applying ReLU activation on float data
__global__ void applyActivation(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f; // ReLU
    }
}


__global__ void calculateBiasGradientKernel(float* layerErrors, float* biasGradient, int layerSize, int batchSize, float* debugArray) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < layerSize) {
        float gradient = 0.0;
        for (int i = 0; i < batchSize; ++i) {
            gradient += layerErrors[i * layerSize + index];
        }
        biasGradient[index] = gradient / batchSize;

        // Store some debug information (e.g., gradient for the first element)
        if (index == 0) {
            debugArray[0] = gradient;
        }
    }
}

// Define CUDA kernel for updating weights and biases
__global__ void updateWeightsBiasesKernel(float* weightGradients, float* biases, float learningRate, int layerSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < layerSize) {
        // Debugging: print values before updating
        if (index == 0) { // Change this condition to control the amount of debugging output
            printf("Thread %d, Bias before update: %f, Weight Gradient: %f\n", index, biases[index], weightGradients[index]);
        }

        // Update bias
        biases[index] -= learningRate * weightGradients[index];

        // Debugging: print values after updating
        if (index == 0) {
            printf("Thread %d, Bias after update: %f\n", index, biases[index]);
        }
    }
}



__global__ void transposeKernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// CUDA Kernel for applying ReLU activation on a batch of data
__global__ void applyBatchReLU(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = max(0.0f, data[idx]); // ReLU
    }
}

// Define the missing function or kernel
__global__ void applyBatchReLUDerivative(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Implement the derivative of ReLU function here
    }
}

// CUDA Kernel for applying the derivative of ReLU on float data
__global__ void applyActivationDerivative(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0.0f ? 1.0f : 0.0f; // Derivative of ReLU
    }
}

__global__ void updateBiasesKernel(float* d_biasGradient, float* d_biases, int layerOutputSize, float alpha) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < layerOutputSize) {
        // Print the index and values for debugging
        printf("Thread index: %d, Original Bias: %f, Bias Gradient: %f\n", idx, d_biases[idx], d_biasGradient[idx]);
        d_biases[idx] += alpha * d_biasGradient[idx];
        printf("Updated Bias: %f\n", d_biases[idx]);
    }
}

// Update the function signature to accept float* pointers
void updateBiases(float* d_biasGradient, float* d_biases, int layerOutputSize, float alpha) {
    // Calculate grid and block dimensions based on layerOutputSize
    int threadsPerBlock = 256;
    int blocksPerGrid = (layerOutputSize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    updateBiasesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_biasGradient, d_biases, layerOutputSize, alpha);

    // Synchronize to ensure that the kernel has completed
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
cudaDeviceSynchronize();
cudaError_t preAllocationError = cudaGetLastError();
if (preAllocationError != cudaSuccess) {
    std::cerr << "Error after updating biases " << cudaGetErrorString(preAllocationError) << std::endl;
    // Handle the error appropriately
}
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;

}


// Simple CPU-based matrix-vector multiplication
std::vector<float> matrixVectorMultiply(const std::vector<float>& matrix, const std::vector<float>& vector, int rows, int cols) {
    std::vector<float> result(rows, 0.0f);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
    return result;
}

// Function to print CUBLAS error message
const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "Unknown CUBLAS error";
    }
}

// Utility function to check CUDA calls
#define CUDA_CHECK(call) \
    do { \
        if((call) != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// Utility function to check cuBLAS calls
#define CUBLAS_CHECK(call) \
    do { \
        if((call) != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CUDA_CHECK_ERROR(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CUBLAS_CHECK_ERROR(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %d (at %s:%d)\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define checkCudaErrors(cudaCall) \
do { \
    cudaError_t cudaError = cudaCall; \
    if (cudaError != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(cudaError) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
} while (0)

#define checkCublasErrors(cublasCall) \
do { \
    cublasStatus_t cublasStatus = cublasCall; \
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << cublasStatus << std::endl; \
        throw std::runtime_error("cuBLAS error"); \
    } \
} while (0)


// Function for matrix multiplication using CuBLAS
void MultiplyMatricesCuBLAS(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int rows_A, int cols_A, int cols_B) {
    // Check dimensions for matrix multiplication validity
    if (A.size() != rows_A * cols_A || B.size() != cols_A * cols_B || C.size() != rows_A * cols_B) {
        std::cerr << "Invalid matrix dimensions." << std::endl;
        return;
    }

    // Initialize cuBLAS handle
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(float) * rows_A * cols_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeof(float) * cols_A * cols_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeof(float) * rows_A * cols_B));

    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(float) * rows_A * cols_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(float) * cols_A * cols_B, cudaMemcpyHostToDevice));

    // Define constants for single precision
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication using cuBLAS
    CUBLAS_CHECK(cublasSgemm(cublasHandle, 
                             CUBLAS_OP_N, CUBLAS_OP_N, 
                             cols_B, rows_A, cols_A, 
                             &alpha, 
                             d_B, cols_B, 
                             d_A, cols_A, 
                             &beta, 
                             d_C, cols_B));

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, sizeof(float) * rows_A * cols_B, cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(cublasHandle);
}

// Function for applying the derivative of the activation function (ReLU in this case)
void applyDerivative(std::vector<float>& data) {
    float *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, data.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (data.size() + blockSize - 1) / blockSize;
    applyActivationDerivative<<<numBlocks, blockSize>>>(d_data, data.size());

    CUDA_CHECK(cudaMemcpy(data.data(), d_data, data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data);
}

// Neural Network class
class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2, int hiddenSize3, int hiddenSize4, int outputSize, float learningRate, int batchSize)

        : inputSize(inputSize), hiddenSize1(hiddenSize1), hiddenSize2(hiddenSize2), hiddenSize3(hiddenSize3), hiddenSize4(hiddenSize4), outputSize(outputSize), learningRate(learningRate), batchSize(batchSize) 

{



        // Initialize weights and biases for all layers
        initializeWeightsAndBiases();
std::cerr << "--------------"<< std::endl;
std::cerr << "Batchsize "<<batchSize << std::endl;
// Initialize the weights vector with your weight vectors
        weights = {
            weightsHidden1,
            weightsHidden2,
            weightsHidden3,
            weightsHidden4,
            weightsOutput
        };

        // Initialize the biases vector with your bias vectors
        biases = {
            biasesHidden1,
            biasesHidden2,
            biasesHidden3,
            biasesHidden4,
            biasesOutput
        };



    }





    void initializeWeightsAndBiases() {
        // Initialize random number generator
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0f, 0.1f);

        // Initialize weights and biases for the hidden layers
        weightsHidden1.resize(inputSize * hiddenSize1);
        biasesHidden1.resize(hiddenSize1);
        for (int i = 0; i < inputSize * hiddenSize1; ++i) {
            weightsHidden1[i] = distribution(generator);
        }
        for (int i = 0; i < hiddenSize1; ++i) {
            biasesHidden1[i] = distribution(generator);
        }

        weightsHidden2.resize(hiddenSize1 * hiddenSize2);
        biasesHidden2.resize(hiddenSize2);
        for (int i = 0; i < hiddenSize1 * hiddenSize2; ++i) {
            weightsHidden2[i] = distribution(generator);
        }
        for (int i = 0; i < hiddenSize2; ++i) {
            biasesHidden2[i] = distribution(generator);
        }

        weightsHidden3.resize(hiddenSize2 * hiddenSize3);
        biasesHidden3.resize(hiddenSize3);
        for (int i = 0; i < hiddenSize2 * hiddenSize3; ++i) {
            weightsHidden3[i] = distribution(generator);
        }
        for (int i = 0; i < hiddenSize3; ++i) {
            biasesHidden3[i] = distribution(generator);
        }

        weightsHidden4.resize(hiddenSize3 * hiddenSize4);
        biasesHidden4.resize(hiddenSize4);
        for (int i = 0; i < hiddenSize3 * hiddenSize4; ++i) {
            weightsHidden4[i] = distribution(generator);
        }
        for (int i = 0; i < hiddenSize4; ++i) {
            biasesHidden4[i] = distribution(generator);
        }

        // Initialize weights and biases for the output layer
        weightsOutput.resize(hiddenSize4 * outputSize);
        biasesOutput.resize(outputSize);
        for (int i = 0; i < hiddenSize4 * outputSize; ++i) {
            weightsOutput[i] = distribution(generator);
        }
        for (int i = 0; i < outputSize; ++i) {
            biasesOutput[i] = distribution(generator);
        }
    }

   




void updateWeights(const std::vector<std::vector<float>>& newWeights) {
        // Assuming weights is a member variable of type std::vector<std::vector<float>>
        for (size_t i = 0; i < weights.size(); ++i) {
            if (weights[i].size() != newWeights[i].size()) {
                std::cerr << "Weight size mismatch at layer " << i << std::endl;
                return;
            }
            weights[i] = newWeights[i];
        }
    }

    void updateBiases(const std::vector<std::vector<float>>& newBiases) {
        // Assuming biases is a member variable of type std::vector<std::vector<float>>
        for (size_t i = 0; i < biases.size(); ++i) {
            if (biases[i].size() != newBiases[i].size()) {
                std::cerr << "Bias size mismatch at layer " << i << std::endl;
                return;
            }
            biases[i] = newBiases[i];
        }
    }


/*
 std::vector<std::vector<float>> forwardBatchNetwork(const std::vector<std::vector<float>>& batchInput) {
        std::vector<std::vector<float>> weights = {weightsHidden1, weightsHidden2, weightsHidden3, weightsHidden4, weightsOutput};
std::vector<std::vector<float>> biases = {biasesHidden1, biasesHidden2, biasesHidden3, biasesHidden4, biasesOutput};
std::vector<int> layerSizes = {inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize};
    return forwardBatch(batchInput, weights, biases, layerSizes);
}
*/
/*

std::vector<std::vector<std::vector<float>>> forwardBatchNetwork(const std::vector<std::vector<float>>& batchInput) {
    std::vector<std::vector<float>> weights = {weightsHidden1, weightsHidden2, weightsHidden3, weightsHidden4, weightsOutput};
    std::vector<std::vector<float>> biases = {biasesHidden1, biasesHidden2, biasesHidden3, biasesHidden4, biasesOutput};
    std::vector<int> layerSizes = {inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize};
    
    return forwardBatch(batchInput, weights, biases, layerSizes);
}
*/

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>> forwardBatchNetwork(const std::vector<std::vector<float>>& batchInput) {



/*
    std::vector<std::vector<float>> weights = {weightsHidden1, weightsHidden2, weightsHidden3, weightsHidden4, weightsOutput};
    std::vector<std::vector<float>> biases = {biasesHidden1, biasesHidden2, biasesHidden3, biasesHidden4, biasesOutput};
*/




    std::vector<int> layerSizes = {inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize










};
    
    return forwardBatch(batchInput, weights, biases, layerSizes);
}





std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>> forwardBatch(
    const std::vector<std::vector<float>>& batchInput,
    const std::vector<std::vector<float>>& weights,  // Weights for each layer
    const std::vector<std::vector<float>>& biases,   // Biases for each layer
    const std::vector<int>& layerSizes               // Sizes of each layer
){

/*
    std::cout << "Weights and Biases at the beginning:" << std::endl;
    for (int layer = 0; layer < weights.size(); ++layer) {
        std::cout << "Layer " << layer << " weights (Size: " << weights[layer].size() << "): ";
        std::cout << "\nLayer " << layer << " biases (Size: " << biases[layer].size() << "): ";
        std::cout << std::endl;
    }

*/


    // Initialize cuBLAS context



/////////////////////////
//// Timing Pre stuff



 cublasHandle_t handle;
cublasStatus_t status = cublasCreate(&handle);
if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS Initialization failed with error code: " << status << std::endl;
    // Handle error or exit
}


    int batchSize = batchInput.size();
    std::cout << "Batch Size: " << batchSize << std::endl;

    int inputSize = layerSizes[0];

    // Flatten batchInput for efficient GPU processing
    std::vector<float> flatBatchInput(batchSize * inputSize);
    for (int i = 0; i < batchSize; ++i) {
        std::copy(batchInput[i].begin(), batchInput[i].end(), flatBatchInput.begin() + i * inputSize);
    }


size_t memorySize = flatBatchInput.size() * sizeof(float);
std::cout << "Allocating " << memorySize << " bytes (" << memorySize / (1024.0 * 1024.0) << " MB) on the GPU." << std::endl;

size_t freeMemBefore, totalMem, freeMemAfter;
cudaMemGetInfo(&freeMemBefore, &totalMem);
std::cout << "Before Allocation - Free Memory: " << freeMemBefore 
          << ", Total Memory: " << totalMem << std::endl;

    // Allocate and copy flatBatchInput to GPU memory
    float* d_batchInput;
    CUDA_CHECK_ERROR(cudaMalloc(&d_batchInput, flatBatchInput.size() * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_batchInput, flatBatchInput.data(), flatBatchInput.size() * sizeof(float), cudaMemcpyHostToDevice));


cudaMemGetInfo(&freeMemAfter, &totalMem);
std::cout << "After Allocation - Free Memory: " << freeMemAfter 
          << ", Total Memory: " << totalMem << std::endl;


    // Variables for intermediate and output data
    float *d_layerInput, *d_layerOutput;
    d_layerInput = d_batchInput; // Initially, input to first layer is batchInput

    std::vector<std::vector<std::vector<float>>> allLayerOutputs;
    std::vector<std::vector<float>> lastLayerOutput;

    for (int layer = 0; layer < weights.size(); ++layer) {
    int outputSize = layerSizes[layer + 1];

    // Allocate GPU memory for weights, biases, and layer output
    float *d_weights, *d_biases;
    CUDA_CHECK_ERROR(cudaMalloc(&d_weights, weights[layer].size() * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_biases, biases[layer].size() * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_layerOutput, batchSize * outputSize * sizeof(float)));

    // Copy weights and biases to GPU memory
    CUDA_CHECK_ERROR(cudaMemcpy(d_weights, weights[layer].data(), weights[layer].size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_biases, biases[layer].data(), biases[layer].size() * sizeof(float), cudaMemcpyHostToDevice));


/////////////////////////
//// Timing Pre stuff










    // Perform matrix multiplication for this layer: d_layerOutput = d_batchInput * d_weights
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   outputSize, batchSize, inputSize,
                                   &alpha,
                                   d_weights, outputSize,
                                   d_layerInput, inputSize,
                                   &beta,
                                   d_layerOutput, outputSize));







    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemm failed" << std::endl;
        exit(EXIT_FAILURE);
    }

   


        // Debugging: Check output size before cudaMemcpy
                std::cout << "Debug: Layer " << layer << " - d_layerOutput size (expected): " << batchSize * outputSize << std::endl;

std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
cudaDeviceSynchronize();
cudaError_t preAllocationError = cudaGetLastError();
if (preAllocationError != cudaSuccess) {
    std::cerr << "CUDA error after sgemm in forwardbatch: " << cudaGetErrorString(preAllocationError) << std::endl;
    // Handle the error appropriately
}
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;



// Apply ReLU activation function
        int numBlocks = (batchSize * outputSize + 255) / 256;
        applyBatchReLU<<<numBlocks, 256>>>(d_layerOutput, batchSize * outputSize);


/*


// Dropout starts here

// Example: Applying dropout to d_layerOutput after activation
float dropoutRate = 0.2; // Example dropout rate
int totalElements = batchSize * outputSize; // Total number of activations in the layer

// Generate dropout mask and apply it
curandGenerator_t curandGen;
CURAND_CHECK_ERROR(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
CURAND_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(curandGen, 123ULL)); // Example seed

// Allocate memory for the dropout mask on the GPU
float* d_dropoutMask;
CUDA_CHECK_ERROR(cudaMalloc(&d_dropoutMask, totalElements * sizeof(float)));

// Generate random numbers for dropout mask
CURAND_CHECK_ERROR(curandGenerateUniform(curandGen, d_dropoutMask, totalElements));

// Kernel to apply dropout mask

// Calculate grid and block sizes
int blockSize = 256; // Example block size, can be tuned
 numBlocks = (totalElements + blockSize - 1) / blockSize;

// Apply dropout mask to d_layerOutput
applyDropout<<<numBlocks, blockSize>>>(d_layerOutput, d_dropoutMask, dropoutRate, totalElements);

// Clean up
CUDA_CHECK_ERROR(cudaFree(d_dropoutMask));
CURAND_CHECK_ERROR(curandDestroyGenerator(curandGen));


*/























         // Copy the processed layer output to host memory
    std::vector<float> layerOutput(batchSize * outputSize);
    CUDA_CHECK_ERROR(cudaMemcpy(layerOutput.data(), d_layerOutput, layerOutput.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Reshape flat layer output to 2D vector
    std::vector<std::vector<float>> reshapedLayerOutput(batchSize, std::vector<float>(outputSize));
    for (int i = 0; i < batchSize; ++i) {
        std::copy(layerOutput.begin() + i * outputSize, layerOutput.begin() + (i + 1) * outputSize, reshapedLayerOutput[i].begin());
    }

    // Store the reshaped output for this layer
    allLayerOutputs.push_back(reshapedLayerOutput);



if (layer == weights.size() - 1) {
            lastLayerOutput = reshapedLayerOutput;
        }

        std::cout << "Layer " << layer << " Output Size: " << allLayerOutputs[layer].front().size() << std::endl;


    // Free memory for weights and biases of this layer
    CUDA_CHECK_ERROR(cudaFree(d_weights));
    CUDA_CHECK_ERROR(cudaFree(d_biases));

    // Free previous layer's input memory and point to the current layer's output
    if (layer > 0) CUDA_CHECK_ERROR(cudaFree(d_layerInput));
    d_layerInput = d_layerOutput;

    // Update inputSize for next layer
    inputSize = outputSize;
}


////////////

    // Copy the final output from GPU to host memory
    std::vector<float> flatOutput(batchSize * layerSizes.back());
    CUDA_CHECK_ERROR(cudaMemcpy(flatOutput.data(), d_layerOutput, flatOutput.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK_ERROR(cudaFree(d_layerInput)); // d_layerOutput is same as d_layerInput in the last iteration
    CUDA_CHECK_ERROR(cudaFree(d_batchInput));
    CUBLAS_CHECK_ERROR(cublasDestroy(handle));

    // Reshape flat output to 2D vector
    std::vector<std::vector<float>> output(batchSize, std::vector<float>(layerSizes.back()));
    for (int i = 0; i < batchSize; ++i) {
        std::copy(flatOutput.begin() + i * layerSizes.back(), flatOutput.begin() + (i + 1) * layerSizes.back(), output[i].begin());
    }
/*
    std::cout << "Weights and Biases at the end:" << std::endl;
    for (int layer = 0; layer < weights.size(); ++layer) {
        std::cout << "Layer " << layer << " weights (Size: " << weights[layer].size() << "): ";
        std::cout << "\nLayer " << layer << " biases (Size: " << biases[layer].size() << "): ";
        std::cout << std::endl;
    }
*/
/*
std::cout << "Layer Outputs:" << std::endl;
    for (int layer = 0; layer < allLayerOutputs.size(); ++layer) {
        std::cout << "Layer " << layer << " Output:" << std::endl;
        for (int sample = 0; sample < batchSize; ++sample) {
            std::cout << "Sample " << sample << ": ";
            for (int neuron = 0; neuron < layerSizes[layer + 1]; ++neuron) {
                std::cout << allLayerOutputs[layer][sample][neuron] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
*/



    return std::make_pair(allLayerOutputs, lastLayerOutput);


}


///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////


void backpropagateBatch(
    const std::vector<std::vector<float>>& batchInput,
    const std::vector<std::vector<float>>& batchTarget,
    //const std::vector<std::vector<float>>& batchOutput, // Output from forwardBatch
    const std::vector<std::vector<std::vector<float>>>& batchOutput, // Outputs from forwardBatch

    float learningRate
)
 {
  
size_t freeMemory, totalMemory;


/////////////////
// Initialization Begin
std::cout << "--------------------------" << std::endl;
std::cout << "--------------------------" << std::endl;
std::cout << "--------------------------" << std::endl;
std::cout << "Backpropagate Batch called" << std::endl;

/*
std::cout << "First few elements of train_targets:" << std::endl;
int numElementsToPrint = std::min(10, 1000); // Change this number to print more or fewer elements
for (int i = 0; i < numElementsToPrint; ++i) {
    std::cout << "train_targets[" << i << "][0]: " << batchTarget[i][0] << std::endl;
}
*/

/*
        std::cerr <<inputSize<<" inputSize"<< std::endl;
        std::cerr <<hiddenSize1<<" hiddenSize1" <<std::endl;
        std::cerr <<hiddenSize2<<" hiddenSize2" <<std::endl;
        std::cerr <<hiddenSize3<<" hiddenSize3"<< std::endl;
        std::cerr <<hiddenSize4<<" hiddenSize4"<< std::endl;
        std::cerr <<outputSize<<" outputSize"<< std::endl;

    const int inputSize = 14;  // Assuming NUM_FEATURES is defined elsewhere
    const int hiddenSize1 = 30;
    const int hiddenSize2 = 15;
    const int hiddenSize3 = 7;
    const int hiddenSize4 = 3;
    const int outputSize = 1;
*/
    std::vector<int> layerSizes = {inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize};
    int numLayers = layerSizes.size();

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Assuming weights and biases are correctly initialized and populated
    std::vector<std::vector<float>> weights = {weightsHidden1, weightsHidden2, weightsHidden3, weightsHidden4, weightsOutput};
    std::vector<std::vector<float>> biases = {biasesHidden1, biasesHidden2, biasesHidden3, biasesHidden4, biasesOutput};

    std::vector<float*> d_weights(numLayers - 1), d_biases(numLayers);
    std::vector<float*> d_layerErrors(numLayers), d_weightGradients(numLayers), d_biasGradients(numLayers);

// Initialization End
/////////////////////////////////////////
///  Memory Allocation for Weights, Biases, Layer Outputs, Errors, and Gradients Start 

// Allocate memory for weights, biases, layer outputs, errors, and gradients on GPU
std::vector<float*> d_batchOutputs(numLayers);  // Additional vector for each layer's output

for (int layer = 0; layer < numLayers; ++layer) {
    std::cout << "Layer " << layer << std::endl;

    int layerInputSize = (layer == 0) ? layerSizes[0] : layerSizes[layer - 1];
    int layerOutputSize = layerSizes[layer];
    cudaError_t cudaStatus;

    // Allocate memory for weights (except for the output layer)
    if (layer < numLayers - 1) {
        cudaStatus = cudaMalloc(&d_weights[layer], weights[layer].size() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed for d_weights[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
            // Handle the error appropriately
        }

        cudaStatus = cudaMemcpy(d_weights[layer], weights[layer].data(), weights[layer].size() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpy failed for d_weights[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
            // Handle the error appropriately
        }
    }


cudaDeviceSynchronize();


    // Allocate memory for biases
    
/*
    cudaStatus = cudaMemcpy(d_biases[layer], biases[layer].data(), biases[layer].size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_biases[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately
    }
*/
    // Allocate memory for the layer's output
    int layerBatchOutputSize = batchSize * layerOutputSize;
    cudaStatus = cudaMalloc(&d_batchOutputs[layer], layerBatchOutputSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_batchOutputs[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately
    }

    std::vector<float> flattenedLayerOutput;
    flattenedLayerOutput.reserve(layerBatchOutputSize);
    for (const auto& batch : batchOutput) {
        flattenedLayerOutput.insert(flattenedLayerOutput.end(), batch[layer].begin(), batch[layer].end());
    }

/*
// Print the values of flattenedLayerOutput
for (const auto& value : flattenedLayerOutput) {
    std::cerr << value << " ";
}
std::cerr << std::endl;
*/
    cudaStatus = cudaMemcpy(d_batchOutputs[layer], flattenedLayerOutput.data(), layerBatchOutputSize * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_batchOutputs[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately
    }

// 2. Allocate host-side memory to read back the data
/*
std::vector<float> hostBatchOutputs(layerBatchOutputSize);

// Copy data from device back to host for printing
cudaStatus = cudaMemcpy(hostBatchOutputs.data(), d_batchOutputs[layer], layerBatchOutputSize * sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMemcpy failed for reading back d_batchOutputs[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
    // Handle the error appropriately, possibly with a return or exit if critical
} else {
    // 3. Print the data
    std::cerr << "Batch Outputs for Layer " << layer << ":\n";
    for (size_t i = 0; i < hostBatchOutputs.size(); ++i) {
        std::cerr << hostBatchOutputs[i] << " ";
        // For better readability, you might want to print a newline every N elements
        if ((i + 1) % 10 == 0) std::cout << "\n";
    }
    std::cerr << "\n"; // Ensure there's a newline after the last element
}
*/

/////////////////////////////////



    // Allocate memory for errors and gradients
    cudaStatus = cudaMalloc(&d_layerErrors[layer], batchSize * layerOutputSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_layerErrors[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately
    }

    cudaStatus = cudaMalloc(&d_weightGradients[layer], layerInputSize * layerOutputSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_weightGradients[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately
    }

    cudaStatus = cudaMalloc(&d_biasGradients[layer], layerOutputSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_biasGradients[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately
    }

    // Check GPU memory
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    std::cout << "Free Memory: " << (freeMemory / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Total Memory: " << (totalMemory / 1024.0 / 1024.0) << " MB" << std::endl;

    // Check for errors after memory allocations
    cudaError_t postAllocationError = cudaGetLastError();
    if (postAllocationError != cudaSuccess) {
        std::cerr << "CUDA error after memory allocation: " << cudaGetErrorString(postAllocationError) << std::endl;
        // Handle the error
    }

}

for (int layer = 0; layer < numLayers-1; ++layer) {

    cudaError_t cudaStatus;

cudaStatus = cudaMalloc(&d_biases[layer], biases[layer].size() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for ad_biases[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately
    }
}

///  Memory Allocation for Weights, Biases, Errors, and Gradients End 
/////////////////////////////////////////
/// Initial Error Calculation Start

const float alpha = -1.0f; // Note: Negative alpha for error calculation
const float beta = 0.0f;

// Debug: Print values from the last layer's output
std::vector<float> lastLayerOutputDebug(batchSize * layerSizes.back());
cudaMemcpy(lastLayerOutputDebug.data(), d_batchOutputs[numLayers - 1], lastLayerOutputDebug.size() * sizeof(float), cudaMemcpyDeviceToHost);
std::cout << "Last layer output (first few values):" << std::endl;
for (int i = 0; i < std::min(10, batchSize * layerSizes.back()); ++i) {
    std::cout << lastLayerOutputDebug[i] << " ";
}
std::cout << std::endl;

std::cout << "Checking dimensions of data structures:" << std::endl;

// Debug: Print dimensions of data structures
std::cout << "batchInput dimensions: " << batchInput.size() << " x " << (batchInput.empty() ? 0 : batchInput[0].size()) << std::endl;
std::cout << "batchTarget dimensions: " << batchTarget.size() << " x " << (batchTarget.empty() ? 0 : batchTarget[0].size()) << std::endl;
std::cout << "batchOutput dimensions: " << batchOutput.size() << " x " << (batchOutput.empty() ? 0 : batchOutput[0][0].size()) << std::endl;

// Debug: Check alpha and beta values
std::cout << "Alpha (for error calculation): " << alpha << std::endl;
std::cout << "Beta: " << beta << std::endl;


// Allocate memory for the initial error calculation
float* d_outputError;
cudaMalloc(&d_outputError, batchSize * layerSizes.back() * sizeof(float));
// Debug: Check if cudaMalloc for d_outputError was successful
std::cout << "Allocated memory for d_outputError: " << batchSize * layerSizes.back() * sizeof(float) << " bytes" << std::endl;

// Copy batchTarget to the GPU
float* d_batchTarget;
checkCuda(cudaMalloc(&d_batchTarget, batchSize * layerSizes.back() * sizeof(float)));
// Debug: Check if cudaMalloc for d_batchTarget was successful
std::cout << "Allocated memory for d_batchTarget: " << batchSize * layerSizes.back() * sizeof(float) << " bytes" << std::endl;

for (int i = 0; i < batchSize; ++i) {
    checkCuda(cudaMemcpy(d_batchTarget + i * layerSizes.back(), batchTarget[i].data(),
                         layerSizes.back() * sizeof(float), cudaMemcpyHostToDevice));
}

// Debug: Print values from batchTarget
std::vector<float> batchTargetDebug(batchSize * layerSizes.back());
cudaMemcpy(batchTargetDebug.data(), d_batchTarget, batchTargetDebug.size() * sizeof(float), cudaMemcpyDeviceToHost);

std::cout << "Batch target (first few values):" << std::endl;
for (int i = 0; i < std::min(10, batchSize * layerSizes.back()); ++i) {
    std::cout << batchTargetDebug[i] << " ";
}
std::cout << std::endl;

// Debug: Confirm batchTarget copied to GPU
std::cout << "Copied batchTarget data to GPU." << std::endl;

std::cout << "Debug: Parameters for cublasSaxpy:" << std::endl;
std::cout << "n: " << batchSize * layerSizes.back() << std::endl;
std::cout << "alpha: " << alpha << std::endl;
std::cout << "d_batchTarget address: " << d_batchTarget << std::endl;
std::cout << "d_batchOutputs[numLayers - 1] address: " << d_batchOutputs[numLayers - 1] << std::endl;

if (d_batchTarget == nullptr) {
    std::cerr << "d_batchTarget is null." << std::endl;
}
if (d_batchOutputs[numLayers - 1] == nullptr) {
    std::cerr << "d_batchOutputs[numLayers - 1] is null." << std::endl;
}

if (handle == nullptr) {
    std::cerr << "cuBLAS handle is not initialized." << std::endl;
}

// Calculate initial error: d_outputError = last layer's output - d_batchTarget
// Using the output of the last layer
cublasStatus_t saxpyStatus = cublasSaxpy(handle, batchSize * layerSizes.back(), &alpha, d_batchTarget, 1, d_batchOutputs[numLayers - 1], 1);

if (saxpyStatus != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasSaxpy failed with status: " << saxpyStatus << std::endl;
    // Handle error appropriately
} else {
    std::cout << "cublasSaxpy executed successfully." << std::endl;
}

// Debug: Print values from d_outputError
std::vector<float> outputErrorDebug(batchSize * layerSizes.back());

cudaMemcpy(outputErrorDebug.data(), d_batchOutputs[numLayers - 1], outputErrorDebug.size() * sizeof(float), cudaMemcpyDeviceToHost);
std::cout << "Initial error (first few values):" << std::endl;
for (int i = 0; i < std::min(10, batchSize * layerSizes.back()); ++i) {
    std::cout << outputErrorDebug[i] << " ";
}
std::cout << std::endl;

//cudaMemcpy(d_outputError, d_batchOutputs[numLayers - 1], batchSize * layerSizes.back() * sizeof(float), cudaMemcpyDeviceToDevice);

// Print dimensions of d_outputError and the last layer's output
std::cout << "Dimensions of d_outputError: " << batchSize << " x " << layerSizes.back() << std::endl;
std::cout << "Dimensions of last layer's output (d_batchOutputs[numLayers - 1]): " << batchSize << " x " << layerSizes.back() << std::endl;

// Allocate host memory for outputError
std::vector<float> outputError(batchSize * layerSizes.back());

// Copy data from GPU to host
checkCuda(cudaMemcpy(outputError.data(), d_batchOutputs[numLayers - 1], outputError.size() * sizeof(float), cudaMemcpyDeviceToHost));

// Optionally print the initial error

/*
std::cout << "Initial Error:" << std::endl;
for (int i = 0; i < batchSize; ++i) {
    for (int j = 0; j < layerSizes.back(); ++j) {
        std::cout << "Row " << i << ", Element " << j << " error: " << outputError[i * layerSizes.back() + j] << std::endl;
    }
}
*/

// Free allocated memory
cudaFree(d_batchTarget);
cudaFree(d_outputError);

/// Initial Error Calculation End
/////////////////////////////////////////
/// Memory Allocation and Transposition for Inputs Start

// Assume batchSize and inputSize are defined and correct
std::cout << "batchSize: " << batchSize << ", inputSize: " << inputSize << std::endl;

// Allocate memory for the transposed input matrix
float* d_transposedInput;
checkCuda(cudaMalloc(&d_transposedInput, batchSize * inputSize * sizeof(float)));

// Copy batchInput to GPU
float* d_batchInputRaw;
checkCuda(cudaMalloc(&d_batchInputRaw, batchSize * inputSize * sizeof(float)));

// Flatten the batch input for copying
std::vector<float> flattenedBatchInput;
for (const auto& inputRow : batchInput) {
    flattenedBatchInput.insert(flattenedBatchInput.end(), inputRow.begin(), inputRow.end());
}
checkCuda(cudaMemcpy(d_batchInputRaw, flattenedBatchInput.data(), batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));

// Debug prints
std::cout << "Debug: Size of flattenedBatchInput: " << flattenedBatchInput.size() << std::endl;
std::cout << "Debug: Size of d_batchInputRaw after copy: " << batchSize * inputSize << std::endl;


// Define new alpha and beta for the transposition operation
float transAlpha = 1.0f;
float transBeta = 0.0f;

int rows = batchSize; // Number of rows in the original matrix
int cols = inputSize; // Number of columns in the original matrix

// Validate memory allocations
if (d_batchInputRaw == nullptr || d_transposedInput == nullptr) {
    std::cerr << "Memory not allocated for matrices involved in transposition." << std::endl;
    // Handle the error
}

/*
cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            inputSize, batchSize, // Dimensions for the transposed matrix
            &alpha,
            d_batchInputRaw, batchSize, // batchSize is the leading dimension in the original input
            &beta,
            NULL, inputSize, // No addition, just transposition
            d_transposedInput, inputSize); // inputSize is the leading dimension in the transposed input
*//*
if (handle == nullptr) {
    std::cerr << "cuBLAS handle is not initialized." << std::endl;
    // Handle error, e.g., initialize the handle or exit the program
}*/
std::cout << "Dimensions of d_batchInputRaw: " << batchSize << " x " << inputSize << std::endl;
std::cout << "Dimensions of d_transposedInput (after transposition): " << inputSize << " x " << batchSize << std::endl;
std::cout << "Leading dimension of d_batchInputRaw: " << batchSize << std::endl;
std::cout << "Leading dimension of d_transposedInput: " << inputSize << std::endl;

// Assuming memory allocation is done previously


/*
cublasStatus_t status = cublasSgeam(handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    inputSize, batchSize, // Note: Swapped dimensions for transposed matrix
                                    &alpha,
                                    d_batchInputRaw, batchSize, // Leading dimension of A is batchSize
                                    &beta,
                                    NULL, inputSize, // Leading dimension of B is inputSize
                                    d_transposedInput, inputSize); // Leading dimension of B is inputSize

*/




// Allocate memory for the transposed matrix on CPU (dimensions are swapped)
    std::vector<float> h_transposedInput(batchSize * inputSize); // Note the swapped dimensions


// Copy the batchInput matrix from GPU to CPU
std::vector<float> h_batchInput(batchSize * inputSize);

    checkCuda(cudaMemcpy(h_batchInput.data(), d_batchInputRaw, batchSize * inputSize * sizeof(float), cudaMemcpyDeviceToHost));




// Transpose the matrix using the provided function

    transposeMatrixCPU(h_batchInput, h_transposedInput, batchSize, inputSize);



// Copy values from h_transposedInput to d_transposedInput on the GPU
checkCuda(cudaMemcpy(d_transposedInput, h_transposedInput.data(), batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));






















/*
cublasStatus_t status = cublasSgeam(handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    batchSize, batchSize, // Output dimensions
                                    &transAlpha,
                                    d_batchInputRaw, batchSize, // Input A and its leading dimension
                                    &transBeta,
                                    NULL, batchSize, // Input B (not used) and its leading dimension
                                    d_transposedInput, batchSize); // Output C and its leading dimension
*/
/*

// Step 1: Allocate host memory for d_batchInputRaw
std::vector<float> h_batchInputRaw(batchSize * batchSize);

// Step 2: Copy data from device to host
cudaError_t cudaStatus2 = cudaMemcpy(h_batchInputRaw.data(), d_batchInputRaw, batchSize * batchSize * sizeof(float), cudaMemcpyDeviceToHost);
if (cudaStatus2 != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus2) << std::endl;
    // Handle error
}

// Step 3: Print the matrix
std::cerr << "Matrix d_batchInputRaw:" << std::endl;
for (int i = 0; i < batchSize; ++i) {
    for (int j = 0; j < batchSize; ++j) {
        std::cerr << h_batchInputRaw[i * batchSize + j] << " ";
    }
    std::cerr << std::endl;
}

*/








 // Leading dimension for the transposed matrix is the number of columns

// Debug print
std::cout << "Debug: Size of d_transposedInput after transposition: " << batchSize * inputSize << std::endl;

// Check for errors after cublasSgeam
cudaError_t cublasError = cudaGetLastError();
if (cublasError != cudaSuccess) {
    std::cerr << "CUDA error after cublasSgeam: " << cudaGetErrorString(cublasError) << std::endl;
    // Handle the error
} else {
    std::cout << "cublasSgeam executed successfully." << std::endl;
}

cudaDeviceSynchronize();
cudaFree(d_batchInputRaw);


//std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
cudaDeviceSynchronize();
cudaError_t preAllocationError = cudaGetLastError();
if (preAllocationError != cudaSuccess) {
    std::cerr << "CUDA error after sgeam immediately in backpropagate batch: " << cudaGetErrorString(preAllocationError) << std::endl;
    // Handle the error appropriately
return;
}
/*
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;


std::cout << "Syncing right now "  << std::endl;
*/
/// Memory Allocation and Transposition for Inputs End
/////////////////
/// Memory Allocation for Transposed Weights Start
// Allocate memory for transposed weights
std::vector<float*> d_weightsTransposed(numLayers - 1);
for (int i = 0; i < numLayers - 1; ++i) {
    int weightsSize = layerSizes[i] * layerSizes[i + 1];
    checkCuda(cudaMalloc(&d_weightsTransposed[i], weightsSize * sizeof(float)));
    
    // Transpose and copy weights to GPU
    // Assuming weights are stored in a flat array in row-major format
    checkCuda(cudaMemcpy(d_weightsTransposed[i], weights[i].data(), weightsSize * sizeof(float), cudaMemcpyHostToDevice));

    // Optionally perform transposition here if required
    // Note: If your weights are already in the correct format for multiplication, you can skip this step
    std::cout << "Debug: Transposed Weights " << i << " - Size after copy: " << weightsSize * sizeof(float) << std::endl;


}

cudaMemGetInfo(&freeMemory, &totalMemory);

std::cout << "GPU Memory Usage:" << std::endl;
std::cout << "Free Memory: " << (freeMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Total Memory: " << (totalMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Used Memory: " << ((totalMemory - freeMemory) / 1024.0 / 1024.0) << " MB" << std::endl;

////////////////////

// Assuming both d_batchOutputs[numLayers - 1] and d_layerErrors[layer] 
// have been allocated with the same size

int numElements = batchSize * layerSizes[5]; // Number of elements to copy
size_t sizeInBytes = numElements * sizeof(float); // Total size in bytes

cudaError_t cudaStatus = cudaMemcpy(d_layerErrors[5], d_batchOutputs[numLayers - 1], sizeInBytes, cudaMemcpyDeviceToDevice);

if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    // Handle the error appropriately

}

/// Memory Allocation for Transposed Weights End

/////////////////////////

 // Example for d_layerErrors
    //std::vector<std::vector<float>> layerErrors(numLayers);


layerErrors.resize(numLayers);

    for (int layer = 0; layer < numLayers; ++layer) {
        int size = batchSize * layerSizes[layer];
        layerErrors[layer].resize(size);
        cudaMemcpy(layerErrors[layer].data(), d_layerErrors[layer], size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_layerErrors[layer]);
    }

std::cout << "Finished copying layererrors" << std::endl;

    // For d_transposedInput
   // std::vector<float> transposedInput(batchSize * inputSize);
 // Finally, destroy cuBLAS handle and synchronize
    cublasDestroy(handle);

transposedInput.resize(batchSize * inputSize);

    cudaMemcpy(transposedInput.data(), d_transposedInput, batchSize * inputSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_transposedInput);

    // For d_batchOutputs
   // std::vector<std::vector<float>> batchOutputs(numLayers);
batchOutputs.resize(numLayers);
    for (int layer = 0; layer < numLayers; ++layer) {
        int size = batchSize * layerSizes[layer];
        batchOutputs[layer].resize(size);
        cudaMemcpy(batchOutputs[layer].data(), d_batchOutputs[layer], size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_batchOutputs[layer]);
    }

 // For d_weightsTransposed
   // std::vector<std::vector<float>> weightsTransposed(numLayers - 1);
 weightsTransposed.resize(numLayers - 1);
    for (int i = 0; i < numLayers - 1; ++i) {
        int size = layerSizes[i] * layerSizes[i + 1];
        weightsTransposed[i].resize(size);
        cudaMemcpy(weightsTransposed[i].data(), d_weightsTransposed[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_weightsTransposed[i]);
    }


    // For d_biasGradients
  //  std::vector<std::vector<float>> biasGradients(numLayers);
biasGradients.resize(numLayers);
    for (int layer = 0; layer < numLayers; ++layer) {
        int size = layerSizes[layer];
        biasGradients[layer].resize(size);
        cudaMemcpy(biasGradients[layer].data(), d_biasGradients[layer], size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_biasGradients[layer]);
    }

    // For d_weightGradients
    //std::vector<std::vector<float>> weightGradients(numLayers);

weightGradients.resize(numLayers - 1);
for (int layer = 0; layer < numLayers - 1; ++layer) {
    int size = layerSizes[layer] * layerSizes[layer + 1];
    weightGradients[layer].resize(size);

    cudaStatus = cudaMalloc(&d_weightGradients[layer], size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_weightGradients[" << layer << "]: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately
    }

    // Copy data from device to host
    cudaMemcpy(weightGradients[layer].data(), d_weightGradients[layer], size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_weightGradients[layer]);
}

////////////////////////////////
/////////////////////////////////

std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
cudaDeviceSynchronize();
cudaError_t preAllocationError2 = cudaGetLastError();
if (preAllocationError2 != cudaSuccess) {
    std::cerr << "CUDA error after sgeam in backpropagatebatch: " << cudaGetErrorString(preAllocationError2) << std::endl;
    // Handle the error appropriately
}
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;



}


///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////

void backpropagateloop(
    const std::vector<std::vector<float>>& batchInput,
    const std::vector<std::vector<float>>& batchTarget,
    const std::vector<std::vector<std::vector<float>>>& batchOutput,
    float learningRate



)
{

auto startCpuTime = std::chrono::high_resolution_clock::now();

std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
cudaDeviceSynchronize();
cudaError_t preAllocationError = cudaGetLastError();
if (preAllocationError != cudaSuccess) {
    std::cerr << "CUDA error at the beginning of backpropagate loop: " << cudaGetErrorString(preAllocationError) << std::endl;
    // Handle the error appropriately
}
std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;

//std::vector<std::vector<float>> weights = {weightsHidden1, weightsHidden2, weightsHidden3, weightsHidden4, weightsOutput};

size_t freeMemory, totalMemory;
cudaMemGetInfo(&freeMemory, &totalMemory);

int numLayers = 6;


std::vector<int> layerSizes = {inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize};

///////////////////////////////
/// Allocating and assigning stuff

 // Error handling variable
    cudaError_t cudaStatus;

    // Initialize cuBLAS context
    cublasHandle_t handle;
    cublasStatus_t cublasStatus = cublasCreate(&handle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle." << std::endl;
        //goto Error;
    }

std::vector<float*> d_weights(numLayers - 1);
size_t totalMemoryAllocated = 0;

for (int layer = 0; layer < numLayers - 1; ++layer) {

std::cout << "layer"<<layer << std::endl;
    int size = layerSizes[layer] * batchSize;//layerSizes[layer + 1];
    size_t memorySize = size * sizeof(float);
    std::cout << "Allocated memory for d_weights[" << layer << "]: " << memorySize << " bytes" << std::endl;
    cudaMalloc(&d_weights[layer], memorySize);
    cudaMemcpy(d_weights[layer], weights[layer].data(), memorySize, cudaMemcpyHostToDevice);
    totalMemoryAllocated += memorySize;
}


std::vector<float*> d_weightsTransposed(numLayers - 1);
for (int i = 0; i < numLayers - 1; ++i) {
    int size = layerSizes[i] * layerSizes[i + 1];
    cudaMalloc(&d_weightsTransposed[i], size * sizeof(float));
    cudaMemcpy(d_weightsTransposed[i], weightsTransposed[i].data(), size * sizeof(float), cudaMemcpyHostToDevice);
}

std::vector<float*> d_batchOutputs(numLayers);
for (int layer = 0; layer < numLayers; ++layer) {
    int size = batchSize * layerSizes[layer];
    cudaMalloc(&d_batchOutputs[layer], size * sizeof(float));
    cudaMemcpy(d_batchOutputs[layer], batchOutputs[layer].data(), size * sizeof(float), cudaMemcpyHostToDevice);
}

// Iterate through the vector and replace NaN values
for (int layer = 0; layer < numLayers; ++layer) {
    int size = batchSize * layerSizes[layer];
    
    // Access the device pointer for the current layer
    float* d_layerOutput = d_batchOutputs[layer];
    
    // Copy the data from device to host for processing
    std::vector<float> h_layerOutput(size);
    cudaMemcpy(h_layerOutput.data(), d_layerOutput, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Replace NaN values in the host data
    for (int i = 0; i < size; ++i) {
        h_layerOutput[i] = replaceNaN(h_layerOutput[i]);
    }
    
    // Copy the updated data back to the device
    cudaMemcpy(d_layerOutput, h_layerOutput.data(), size * sizeof(float), cudaMemcpyHostToDevice);
}

// Declare the vector of device pointers
std::vector<float*> d_layerErrors(numLayers);

// Allocate memory and copy data for each layer
for (int layer = 0; layer < numLayers; ++layer) {
    int size = batchSize * layerSizes[layer];
    float* d_layerError;
    cudaMalloc(&d_layerError, size * sizeof(float));
    cudaMemcpy(d_layerError, layerErrors[layer].data(), size * sizeof(float), cudaMemcpyHostToDevice);
    d_layerErrors[layer] = d_layerError;
}

// Sync to ensure all operations are done
cudaDeviceSynchronize();

float* d_transposedInput;
int transposedInputSize = batchSize * inputSize; // Adjust according to your input size
cudaMalloc(&d_transposedInput, transposedInputSize * sizeof(float));
cudaMemcpy(d_transposedInput, transposedInput.data(), transposedInputSize * sizeof(float), cudaMemcpyHostToDevice);

std::vector<float*> d_weightGradients(numLayers - 1); // No weights for the input layer
std::vector<float*> d_biasGradients(numLayers);

for (int layer = 0; layer < numLayers; ++layer) {
    // Allocate bias gradients
    int biasSize = layerSizes[layer];
    cudaMalloc(&d_biasGradients[layer], biasSize * sizeof(float));
    cudaMemset(d_biasGradients[layer], 0, biasSize * sizeof(float)); // Initialize to zero

    if (layer > 0) {
        // Allocate weight gradients
        int weightSize = layerSizes[layer] * layerSizes[layer - 1]; // layerSizes[layer-1] is the size of the previous layer
        cudaMalloc(&d_weightGradients[layer - 1], weightSize * sizeof(float));
        cudaMemset(d_weightGradients[layer - 1], 0, weightSize * sizeof(float)); // Initialize to zero
    }
}

std::cout << "Biases (CPU):" << std::endl;
for (int layer = 0; layer < biases.size(); ++layer) {
    std::cout << "Layer " << layer << " biases:" << std::endl;
    std::cout << "Size: " << biases[layer].size() << " | ";

    // Print the first few elements of each biases vector
    std::cout << "First elements: ";
    for (int i = 0; i < std::min(static_cast<int>(biases[layer].size()), 2); ++i) {
        std::cout << biases[layer][i] << " ";
    }
    std::cout << std::endl;
}

size_t freeMemoryBefore, totalMemory2;
cudaError_t memoryInfoStatus = cudaMemGetInfo(&freeMemoryBefore, &totalMemory2);
if (memoryInfoStatus != cudaSuccess) {
    std::cerr << "Error getting CUDA memory info: " << cudaGetErrorString(memoryInfoStatus) << std::endl;
    // Handle the error as needed
}

size_t usedMemoryBefore = totalMemory2 - freeMemoryBefore;
std::cout << "Free Memory before bias allocation: " << (freeMemoryBefore / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Used Memory before bias allocation: " << (usedMemoryBefore / 1024.0 / 1024.0) << " MB" << std::endl;

std::vector<float*> d_biases(numLayers);
for (int layer = 0; layer < numLayers-1; ++layer) {
    int size = layerSizes[layer];
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&d_biases[layer], size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_biases[" << layer << "] with error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately (e.g., cleanup and error propagation)
        continue; // Or other error handling
    }

    cudaStatus = cudaMemcpy(d_biases[layer], biases[layer].data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_biases[" << layer << "] with error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle the error appropriately
        continue; // Or other error handling
    }
}

std::cout << "Biases copied to GPU successfully." << std::endl;

std::cout << "After  Use transposed weights for backpropagation" << std::endl;

  float alpha = 1.0;

  float beta = 1.0;


cudaMemGetInfo(&freeMemory, &totalMemory);

std::cout << "GPU Memory Usage After all the allocats:" << std::endl;
std::cout << "Free Memory: " << (freeMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Total Memory: " << (totalMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Used Memory: " << ((totalMemory - freeMemory) / 1024.0 / 1024.0) << " MB" << std::endl;

float* d_biasGradient;

///////////////////////////////

// Assuming batchOutput[outputLayerIndex] and batchTarget are std::vector<std::vector<float>>

int outputLayerIndex = numLayers - 1;
int outputSize = layerSizes[outputLayerIndex];
std::vector<float> outputLayerError(batchSize * outputSize);
/*
std::cout << "Output Layer Index: " << outputLayerIndex << std::endl;
std::cout << "Output Size: " << outputSize << std::endl;
std::cout << "Batch Size: " << batchSize << std::endl;
*/

// Calculate MSE error on CPU
for (int i = 0; i < batchSize; ++i) {
    for (int j = 0; j < outputSize; ++j) {
        float predicted = batchOutput[outputLayerIndex-1][i][j];
        float actual = batchTarget[i][j];
        //std::cout << "Predicted: " << predicted << ", Actual: " << actual  << std::endl;
        outputLayerError[i * outputSize + j] = 2.0f * (predicted - actual) / static_cast<float>(batchSize);
    }
}

// Copy the calculated error to the device
float* d_outputLayerError;
cudaMalloc(&d_outputLayerError, outputLayerError.size() * sizeof(float));
cudaMemcpy(d_outputLayerError, outputLayerError.data(), outputLayerError.size() * sizeof(float), cudaMemcpyHostToDevice);


float* d_onesVector;
cudaMalloc(&d_onesVector, batchSize * sizeof(float));

std::vector<float> onesVector(batchSize, 1.0f);
cudaMemcpy(d_onesVector, onesVector.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice);
/*
std::cout << "##################################### " << std::endl;
std::cout << "##################################### " << std::endl;
std::cout << "##################################### " <<  std::endl;
std::cout << "Backpropagating " <<  std::endl;
*/
float gradientClippingThreshold = 1.0f; // Example threshold, adjust as needed

 outputLayerIndex = layerSizes.size() - 1; // Last layer (output layer)

// Backpropagate the error to hidden layers (exclude input layer)
for (int layer = numLayers - 1; layer > 1; --layer) {
    int layerInputSize = (layer == 1) ? inputSize : layerSizes[layer - 1]; // inputSize for the first layer
    int layerOutputSize = layerSizes[layer];

int weightSize = layerInputSize * layerOutputSize; // Total number of weights for the current layer

/////////////////////////
//// Timing Cuda CAll

// Initialize CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);


    // Calculate weight gradients for current layer
    // d_weightGradients[layer - 1] = d_layerErrors[layer] (transposed) * d_batchOutputs[layer - 1]
    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                layerOutputSize, layerInputSize, batchSize,
                &alpha, d_layerErrors[layer], layerOutputSize,
                d_batchOutputs[layer - 1], layerInputSize,
                &beta, d_weightGradients[layer - 1], layerOutputSize);


 if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemv for biases failed at Layer " << layer << std::endl;
        // Handle error...
    }

cudaDeviceSynchronize();

/*
// L2 regularization starts here

    float lambda = 0.005; // L2 regularization strength
    float alpha = -learningRate;
    float l2_alpha = -learningRate * lambda; // For L2 regularization term

    // Update gradients to include L2 regularization term
    // This effectively modifies the gradient to include a penalty for large weights
    cublasSaxpy(handle, weightSize, &l2_alpha, d_weights[layer - 1], 1, d_weightGradients[layer - 1], 1);

    // Now, apply the updated gradient to adjust the weights
    cublasSaxpy(handle, weightSize, &alpha, d_weightGradients[layer - 1], 1, d_weights[layer - 1], 1);



cudaDeviceSynchronize();

*/

    // Calculate bias gradients for current layer
    // d_biasGradients[layer - 1] = sum(d_layerErrors[layer])
    cublasStatus = cublasSgemv(handle, CUBLAS_OP_N,
                batchSize, layerOutputSize,
                &alpha, d_layerErrors[layer], batchSize,
                d_onesVector, 1,
                &beta, d_biasGradients[layer - 1], 1);



 if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemv for biases failed at Layer " << layer << std::endl;
        // Handle error...
    }

cudaDeviceSynchronize();


 // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cerr << "Time for cublasSgemm: " << milliseconds << " ms\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


/////////////////////////
//// Timing Cuda CAll

int threadsPerBlock = 256; // You can adjust this based on your GPU capabilities
        int numBlocks = (layerSizes[layer - 1] * batchSize + threadsPerBlock - 1) / threadsPerBlock;
 // Apply gradient clipping to prevent exploding gradients
    applyGradientClipping<<<numBlocks, threadsPerBlock>>>(d_weightGradients[layer - 1], layerInputSize * layerOutputSize, gradientClippingThreshold);
    applyGradientClipping<<<numBlocks, threadsPerBlock>>>(d_biasGradients[layer - 1], layerOutputSize, gradientClippingThreshold);

// Assuming 'layer' is the current layer index
 weightSize = layerInputSize * layerOutputSize; // Size for d_weightGradients
int biasSize = layerOutputSize; // Size for d_biasGradients

    // Propagate the error to the previous layer (if not the first layer)
    if (layer > 0) {

        cublasStatus = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    layerSizes[layer - 1], batchSize, layerOutputSize,
                    &alpha, d_weights[layer - 1], layerOutputSize,
                    d_layerErrors[layer], layerOutputSize,
                    &beta, d_layerErrors[layer - 1], layerSizes[layer - 1]);
        
        // Apply the derivative of the activation function (ReLU in this case)

 if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasSgemv for biases failed at Layer " << layer << std::endl;
        // Handle error...
    }
        dim3 gridSize(numBlocks, 1, 1);
        dim3 blockSize(threadsPerBlock, 1, 1);
        applyReluDerivative2<<<gridSize, blockSize>>>(d_layerErrors[layer - 1], d_batchOutputs[layer - 1], layerSizes[layer - 1] * batchSize);
    }

cudaDeviceSynchronize();

    // Error checking and synchronization (optional but recommended)
    // Check for errors in cuBLAS operations and synchronize the device

///////////////////////////////////////

// Update biases on GPU
float negativeLearningRate = -learningRate;
cublasSaxpy(handle, layerSizes[layer], &negativeLearningRate, d_biasGradients[layer - 1], 1, d_biases[layer - 1], 1);

// Update weights on GPU
cublasSaxpy(handle, layerInputSize * layerOutputSize, &negativeLearningRate, d_weightGradients[layer - 1], 1, d_weights[layer - 1], 1);

cudaError_t err;
err = cudaMemcpy(weights[layer - 1].data(), d_weights[layer - 1], weightSize, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) {
    std::cerr << "CUDA error in layer " << layer << " after copying weights: " << cudaGetErrorString(err) << std::endl;
}

err = cudaMemcpy(biases[layer - 1].data(), d_biases[layer - 1], biasSize, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) {
    std::cerr << "CUDA error in layer " << layer << " after copying biases: " << cudaGetErrorString(err) << std::endl;
}

///////////////////////////
   // std::cerr << "layer: "<<layer<< std::endl;
// Check for errors after memory cleanup

///////////////////////////


//std::cerr << "----------------------------------"<< std::endl;
//std::cerr << "layer " << layer - 1 << " to layer "<<layer<< std::endl;
//std::cerr << "----------------------------------"<< std::endl;

cudaError_t cleanupError2 = cudaGetLastError();
if (cleanupError2 != cudaSuccess) {
    std::cerr << "layer: "<<layer<<" CUDA error after copying biases and weights: " << cudaGetErrorString(cleanupError2) << std::endl;
    // Handle the error
} else {
    std::cout << "Memory cleanup completed successfully." << std::endl;
}

}
/*

std::cout << "##################################### " << std::endl;
std::cout << "##################################### " << std::endl;
std::cout << "##################################### " <<  std::endl;
std::cout << "Backpropagating loop ends here " <<  std::endl;
*/
// Copy updated biases from GPU to CPU

for (int layer = 1; layer < numLayers - 1; ++layer) {
    int size = layerSizes[layer]; // Ensure this is the size used in cudaMalloc for d_biases[layer]

    cudaError_t err = cudaMemcpy(biases[layer].data(), d_biases[layer], size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in layer " << layer << " after copying biases: " << cudaGetErrorString(err) << std::endl;
        // Handle the error appropriately
    }
}

// Release allocated GPU memory
for (int layer = 0; layer < numLayers - 1; ++layer) {
    cudaError_t err;

    err = cudaFree(d_weights[layer]);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in freeing d_weights[" << layer << "]: " << cudaGetErrorString(err) << std::endl;
    }
    d_weights[layer] = nullptr;

    err = cudaFree(d_weightsTransposed[layer]);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in freeing d_weightsTransposed[" << layer << "]: " << cudaGetErrorString(err) << std::endl;
    }
    d_weightsTransposed[layer] = nullptr;

    err = cudaFree(d_weightGradients[layer]);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in freeing d_weightGradients[" << layer << "]: " << cudaGetErrorString(err) << std::endl;
    }
    d_weightGradients[layer] = nullptr;
}

for (int layer = 0; layer < numLayers; ++layer) {
    cudaError_t err;

    err = cudaFree(d_biasGradients[layer]);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in freeing d_biasGradients[" << layer << "]: " << cudaGetErrorString(err) << std::endl;
    }
    d_biasGradients[layer] = nullptr;

    err = cudaFree(d_biases[layer]);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in freeing d_biases[" << layer << "]: " << cudaGetErrorString(err) << std::endl;
    }
    d_biases[layer] = nullptr;

    err = cudaFree(d_batchOutputs[layer]);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in freeing d_batchOutputs[" << layer << "]: " << cudaGetErrorString(err) << std::endl;
    }
    d_batchOutputs[layer] = nullptr;

    err = cudaFree(d_layerErrors[layer]);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in freeing d_layerErrors[" << layer << "]: " << cudaGetErrorString(err) << std::endl;
    }
    d_layerErrors[layer] = nullptr;
}

cudaError_t err;

err = cudaFree(d_transposedInput);
if (err != cudaSuccess) {
    std::cerr << "CUDA error in freeing d_transposedInput: " << cudaGetErrorString(err) << std::endl;
}
d_transposedInput = nullptr;

err = cudaFree(d_outputLayerError);
if (err != cudaSuccess) {
    std::cerr << "CUDA error in freeing d_outputLayerError: " << cudaGetErrorString(err) << std::endl;
}
d_outputLayerError = nullptr;


err = cudaFree(d_onesVector);
if (err != cudaSuccess) {
    std::cerr << "CUDA error in freeing d_onesVector: " << cudaGetErrorString(err) << std::endl;
}

// Destroy the cuBLAS handle
cublasStatus_t cublasStatus2 = cublasDestroy(handle);
if (cublasStatus2 != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Failed to destroy cuBLAS handle." << std::endl;
}


/*

// Check for errors after cublasSgeam
std::cout << " ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
std::cout << " ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
std::cout << " ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
cudaError_t cublasError2 = cudaGetLastError();
if (cublasError2 != cudaSuccess) {
    std::cerr << "CUDA error before cublasSgeam: " << cudaGetErrorString(cublasError2) << std::endl;
    // Handle the error
} else {
    std::cout << "cublasSgeam executed successfully." << std::endl;
}

std::cout << " vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << std::endl;
std::cout << " vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << std::endl;
std::cout << " vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << std::endl;
*/


auto endCpuTime = std::chrono::high_resolution_clock::now();
auto durationCpu = std::chrono::duration_cast<std::chrono::milliseconds>(endCpuTime - startCpuTime);
std::cerr << "Total CPU execution time after: " << durationCpu.count() << " ms\n";


}

///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////


 // Function to update weights and biases
    void updateWeightsAndBiases(const std::vector<float>& input, const std::vector<float>& hidden1, const std::vector<float>& hidden2, const std::vector<float>& hidden3, const std::vector<float>& hidden4,
        const std::vector<float>& outputError, const std::vector<float>& hiddenError4, const std::vector<float>& hiddenError3, const std::vector<float>& hiddenError2, const std::vector<float>& hiddenError1) {

        // Update weights and biases for the output layer
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < hiddenSize4; ++j) {
                weightsOutput[i * hiddenSize4 + j] -= learningRate * outputError[i] * hidden4[j];
            }
            biasesOutput[i] -= learningRate * outputError[i];
        }

        // Update weights and biases for hidden layer 4
        for (int i = 0; i < hiddenSize4; ++i) {
            for (int j = 0; j < hiddenSize3; ++j) {
                weightsHidden4[i * hiddenSize3 + j] -= learningRate * hiddenError4[i] * hidden3[j];
            }
            biasesHidden4[i] -= learningRate * hiddenError4[i];
        }

        // Update weights and biases for hidden layer 3
        for (int i = 0; i < hiddenSize3; ++i) {
            for (int j = 0; j < hiddenSize2; ++j) {
                weightsHidden3[i * hiddenSize2 + j] -= learningRate * hiddenError3[i] * hidden2[j];
            }
            biasesHidden3[i] -= learningRate * hiddenError3[i];
        }

        // Update weights and biases for hidden layer 2
        for (int i = 0; i < hiddenSize2; ++i) {
            for (int j = 0; j < hiddenSize1; ++j) {
                weightsHidden2[i * hiddenSize1 + j] -= learningRate * hiddenError2[i] * hidden1[j];
            }
            biasesHidden2[i] -= learningRate * hiddenError2[i];
        }

        // Update weights and biases for hidden layer 1
        for (int i = 0; i < hiddenSize1; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weightsHidden1[i * inputSize + j] -= learningRate * hiddenError1[i] * input[j];
            }
            biasesHidden1[i] -= learningRate * hiddenError1[i];
        }
    }


bool hasConverged(const std::vector<float>& previousLosses, const float currentLoss, const int patience = 10) {
    // Convergence criteria: Check if the loss has not improved for 'patience' epochs
    if (previousLosses.size() < patience) {
        // If we don't have enough previous losses for comparison, continue training
        return false;
    }

    for (int i = 1; i <= patience; ++i) {
        // Check if the loss has decreased over the last 'patience' epochs
        if (currentLoss >= previousLosses[previousLosses.size() - i]) {
            return true;
        }
    }

    return false;
}


/*
bool hasConverged(const std::vector<float>& previousLosses, const float currentLoss, const int patience = 10) {
    if (previousLosses.empty()) {
        return false; // If there are no previous losses, continue training
    }

    // Check if the last loss in previousLosses is less than the threshold
    if (previousLosses.back() < 0.05) {
        return true; // Training has converged
    }

    return false; // Continue training
}
*/




/*
float calculateLoss(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets) {
    // Ensure that predictions and targets have the same dimensions
    if (predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) {
        std::cerr << "Error: Predictions and targets have mismatched dimensions." << std::endl;
        return 0.0f; // Return a default loss value
    }

    int numSamples = predictions.size();
    int numOutputs = predictions[0].size();

    float totalLoss = 0.0f;


for (int i = 0; i < numSamples; ++i) {
        for (int j = 0; j < numOutputs; ++j) {
            float error = predictions[i][j] - targets[i][j];
            totalLoss += error * error; // Squared error
            // Print every 100th error for monitoring (or adjust this as needed)
            if ((i * numOutputs + j) % 100 == 0) {
                std::cout << "Sample " << i << ", Output " << j << ", Error: " << error << std::endl;
            }
        }
    }
    // Calculate the mean squared error
    float meanSquaredError = totalLoss / (numSamples * numOutputs);
        std::cout << "Calculated loss corrrectly" << std::endl;
    return meanSquaredError;
}
*/

float calculateLoss(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets) {
    // Ensure that predictions and targets have the same dimensions
    if (predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) {
        std::cerr << "Error: Predictions and targets have mismatched dimensions." << std::endl;
        return 0.0f; // Return a default loss value
    }

    int numSamples = predictions.size();
    int numOutputs = predictions[0].size();

    float totalLoss = 0.0f;

    for (int i = 0; i < numSamples; ++i) {
        for (int j = 0; j < numOutputs; ++j) {
            float error = predictions[i][j] - targets[i][j];
            totalLoss += error * error; // Squared error
        }
    }

    // Calculate the mean squared error
    float meanSquaredError = totalLoss / (numSamples * numOutputs);

    // Calculate the root mean squared error (RMSE)
    float rootMeanSquaredError = std::sqrt(meanSquaredError);

    return rootMeanSquaredError;
}



/*
// Train function
void trainNeuralNetwork(
    NeuralNetwork& neuralNetwork,
    std::vector<std::vector<float>>& train_inputs,
    std::vector<std::vector<float>>& train_targets,
    float learningRate,
    int maxEpochs,
    float convergenceThreshold = 0.001
) {

int increasingLossWindowSize = 3; // Number of epochs to check for increasing loss

size_t freeMemory, totalMemory;
cudaMemGetInfo(&freeMemory, &totalMemory);

std::cout << "GPU Memory Usage:" << std::endl;
std::cout << "Free Memory: " << (freeMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Total Memory: " << (totalMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Used Memory: " << ((totalMemory - freeMemory) / 1024.0 / 1024.0) << " MB" << std::endl;


    int batchSize = train_inputs.size();
    std::vector<float> previousLosses;

    std::ofstream lossFile("loss.txt"); // Open a file for writing losses

    for (int epoch = 1; epoch <= maxEpochs; ++epoch) {

cudaMemGetInfo(&freeMemory, &totalMemory);

std::cout << "GPU Memory Usage at epoch start:" << std::endl;
std::cout << "Free Memory: " << (freeMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Total Memory: " << (totalMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Used Memory: " << ((totalMemory - freeMemory) / 1024.0 / 1024.0) << " MB" << std::endl;

std::cerr << "-------------------" << std::endl;
std::cerr << "Epoch"<<epoch << std::endl;

std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;

std::cerr << "Weights (First 20 elements):" << std::endl;
for (const std::vector<float>& layerWeights : neuralNetwork.weights) {
    for (size_t i = 0; i < std::min(layerWeights.size(), static_cast<size_t>(20)); ++i) {
        std::cerr << layerWeights[i] << " ";
    }
    std::cerr << std::endl;
}

std::cerr << "Biases (First 20 elements):" << std::endl;
for (const std::vector<float>& layerBiases : neuralNetwork.biases) {
    for (size_t i = 0; i < std::min(layerBiases.size(), static_cast<size_t>(20)); ++i) {
        std::cerr << layerBiases[i] << " ";
    }
    std::cerr << std::endl;
}

        // Forward pass to compute predictions
        auto forwardResults = forwardBatchNetwork(train_inputs);
        auto predictions = forwardResults.second; // Use the second value of predictions
        
        // Calculate the loss using the modified calculateLoss function
        float loss = calculateLoss(predictions, train_targets);
                std::cout << "loss " << loss << std::endl;

// Store the current loss for future comparison
    previousLosses.push_back(loss);

     // Print current loss
        std::cerr << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        lossFile << "Epoch " << epoch << ", Loss: " << loss << std::endl;

        // Check for convergence
        if (epoch > 1) {
            // If we have more than one epoch, check for convergence
            if (hasConverged(previousLosses, loss, convergenceThreshold)) {
                std::cout << "Convergence achieved. Stopping training." << std::endl;
                break;
            }
        }
        
// Check for increasing loss
    if (areLossesIncreasing(previousLosses, increasingLossWindowSize)) {
        std::cerr << "Losses have been increasing for " << increasingLossWindowSize << " epochs. Stopping training." << std::endl;
        break;
    }


        // Store the current loss for future comparison
        previousLosses.push_back(loss);

        // Backpropagation to update weights and biases
        neuralNetwork.backpropagateBatch(train_inputs, train_targets, forwardResults.first, learningRate);
        neuralNetwork.backpropagateloop(train_inputs, train_targets, forwardResults.first, learningRate);
      

 


cudaMemGetInfo(&freeMemory, &totalMemory);

std::cout << "GPU Memory Usage at epoch end:" << std::endl;
std::cout << "Free Memory: " << (freeMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Total Memory: " << (totalMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Used Memory: " << ((totalMemory - freeMemory) / 1024.0 / 1024.0) << " MB" << std::endl;

    }
    lossFile.close(); // Close the loss file when done

}
*/

/*
void trainNeuralNetwork(
    NeuralNetwork& neuralNetwork,
    const std::vector<std::vector<float>>& train_inputs,
    const std::vector<std::vector<float>>& train_targets,
    const std::vector<std::vector<float>>& test_inputs,
    const std::vector<std::vector<float>>& test_targets,
    float learningRate,
    int maxEpochs,
    float convergenceThreshold = 0.5
) {
    int increasingLossWindowSize = 500; // Number of epochs to check for increasing loss
    std::vector<float> previousLosses;
    std::ofstream lossFile("loss.txt"); // Open a file for writing losses

    for (int epoch = 1; epoch <= maxEpochs; ++epoch) {



cudaError_t cudaStatus = cudaDeviceReset(); // Reset the device

        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaDeviceReset failed for device " <<  ": " << cudaGetErrorString(cudaStatus) << std::endl;
        } else {
            std::cout << "CUDA device " <<  " reset successfully." << std::endl;
        }





        std::cerr << "------------------------------------ " << std::endl;
        std::cerr << "Epoch " << epoch << std::endl;
size_t freeMemory, totalMemory;
cudaMemGetInfo(&freeMemory, &totalMemory);

std::cerr << "GPU Memory Usage at epoch end:" << std::endl;
std::cerr << "Free Memory: " << (freeMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cerr << "Total Memory: " << (totalMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cerr << "Used Memory: " << ((totalMemory - freeMemory) / 1024.0 / 1024.0) << " MB" << std::endl;



std::cerr << "Biases (First 20 elements):" << std::endl;
for (const std::vector<float>& layerBiases : neuralNetwork.biases) {
    for (size_t i = 0; i < std::min(layerBiases.size(), static_cast<size_t>(200)); ++i) {
        std::cerr << layerBiases[i] << " ";
    }
    std::cerr << std::endl;
}

        // Forward pass to compute predictions
        auto forwardResults = forwardBatchNetwork(train_inputs);
        auto predictions = forwardResults.second;

        // Calculate the loss
        float loss = calculateLoss(predictions, train_targets);
        std::cerr << "Loss: " << loss << std::endl;
        lossFile << "Epoch " << epoch << ", Loss: " << loss << std::endl;


// Calculate and print test loss
        auto testOutput = neuralNetwork.forwardBatchNetwork(test_inputs);
        float testLoss = calculateLoss(testOutput.second, test_targets);
        std::cerr << "Test Loss at start of Epoch " << epoch << ": " << testLoss << std::endl;





        // Check for convergence before adding the current loss
        if (epoch > 1 && hasConverged(previousLosses, loss, convergenceThreshold)) {
            std::cout << "Convergence achieved. Stopping training." << std::endl;
            break;
        }

        // Check for increasing loss
        if (areLossesIncreasing(previousLosses, increasingLossWindowSize)) {
            std::cerr << "Losses have been increasing for " << increasingLossWindowSize << " epochs. Stopping training." << std::endl;
            break;
        }

        // Store the current loss for future comparison
        previousLosses.push_back(loss);

        // Backpropagation to update weights and biases
        neuralNetwork.backpropagateBatch(train_inputs, train_targets, forwardResults.first, learningRate);

  neuralNetwork.backpropagateloop(train_inputs, train_targets, forwardResults.first, learningRate);
      

    }

    lossFile.close(); // Close the loss file when done
}

*/

bool trainNeuralNetwork(
    NeuralNetwork& neuralNetwork,
    const std::vector<std::vector<std::vector<float>>>& train_inputs_batches,
    const std::vector<std::vector<std::vector<float>>>& train_targets_batches,
    const std::vector<std::vector<float>>& test_inputs,
    const std::vector<std::vector<float>>& test_targets,
    float learningRate,
    int maxEpochs
) {






// Parameters for learning rate schedule
int stepSize = 10; // Decrease learning rate every 10 epochs
float decayFactor = 0.6f; // Learning rate is multiplied by 0.9 at each step


    std::ofstream lossFile("loss.txt"); // Open a file for writing losses
    int increasingLossWindowSize = 500; // Number of epochs to check for increasing loss
    std::vector<float> previousLosses;
    float lowestRMSE = std::numeric_limits<float>::max();

float previousLoss = std::numeric_limits<float>::max();
float lossChangeThreshold = 0.05f; // Threshold for significant loss change
float adaptiveDecayFactor = 0.95f; // Factor to reduce learning rate by when loss change is below threshold



    for (int epoch = 1; epoch <= maxEpochs; ++epoch) {
 // Start timing
    auto start = std::chrono::high_resolution_clock::now();




cudaError_t cudaStatus = cudaDeviceReset(); // Reset the device

        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaDeviceReset failed for device " <<  ": " << cudaGetErrorString(cudaStatus) << std::endl;
        } else {
            std::cout << "CUDA device " <<  " reset successfully." << std::endl;
        }



bool isUnstableNetwork = false;

// Check weights for NaN
for (const std::vector<float>& layerWeights : neuralNetwork.weights) {
    for (float weight : layerWeights) {
        if (std::isnan(weight)) {
            isUnstableNetwork = true;
            break;
        }
    }
    if (isUnstableNetwork) break;
}

// Check biases for NaN
if (!isUnstableNetwork) { // Only proceed if no NaN found in weights
    for (const std::vector<float>& layerBiases : neuralNetwork.biases) {
        for (float bias : layerBiases) {
            if (std::isnan(bias)) {
                isUnstableNetwork = true;
                break;
            }
        }
        if (isUnstableNetwork) break;
    }
}


// Print warning or biases based on check
if (isUnstableNetwork) {
    std::cerr << "********unstable network********" << std::endl;
return false;
} /*else {
    std::cerr << "Biases (First 20 elements):" << std::endl;
    for (const std::vector<float>& layerBiases : neuralNetwork.biases) {
        for (size_t i = 0; i < std::min(layerBiases.size(), static_cast<size_t>(20)); ++i) {
            std::cerr << layerBiases[i] << " ";
        }
        std::cerr << std::endl;
    }
}*/
/*
std::cerr << "Biases (First 20 elements):" << std::endl;
for (const std::vector<float>& layerBiases : neuralNetwork.biases) {
    for (size_t i = 0; i < std::min(layerBiases.size(), static_cast<size_t>(200)); ++i) {
        std::cerr << layerBiases[i] << " ";
    }
    std::cerr << std::endl;
}
*/

        float epochLoss = 0.0f;
        //std::cerr << "Epoch " << epoch << std::endl;
        for (size_t batchIndex = 0; batchIndex < train_inputs_batches.size(); ++batchIndex) {
            auto forwardResults = neuralNetwork.forwardBatchNetwork(train_inputs_batches[batchIndex]);
            


neuralNetwork.backpropagateBatch(train_inputs_batches[batchIndex], train_targets_batches[batchIndex], forwardResults.first, learningRate);

neuralNetwork.backpropagateloop(train_inputs_batches[batchIndex], train_targets_batches[batchIndex], forwardResults.first, learningRate);




            float batchLoss = calculateLoss(forwardResults.second, train_targets_batches[batchIndex]);
            epochLoss += batchLoss;
        }

        epochLoss /= train_inputs_batches.size();

float averageEpochLoss = epochLoss ;

        std::cerr << "Epoch " << epoch << ", Average Loss: " << epochLoss << std::endl;



// Adaptive learning rate here


if (epoch % stepSize == 0) {
        learningRate *= decayFactor;
        std::cerr << "Learning rate decreased to " << learningRate << std::endl;
    }

/*
if (epochLoss<0.15f){learningRate*= 0.55f;        std::cerr << "Learning rate 1.1 decreased to " << learningRate << std::endl;}
*/

/*
// Calculate the absolute change in loss and the threshold for comparison
float lossChange = std::abs(previousLoss - averageEpochLoss);
float thresholdForChange = lossChangeThreshold * previousLoss;

std::cerr << "Loss Change: " << lossChange << ", Threshold for Significant Change: " << thresholdForChange << std::endl;

// Check if the change in loss is less than the threshold
if (lossChange < thresholdForChange) {
    // Reduce learning rate if the change in loss is not significant
    learningRate *= adaptiveDecayFactor;
    std::cerr << "Adaptive learning rate adjustment: New learning rate is " << learningRate << std::endl;
    std::cerr << "Condition Met: YES - Learning rate adjusted." << std::endl;
} else {
    std::cerr << "Condition Met: NO - Learning rate remains the same." << std::endl;
}

// Update previous loss for next epoch comparison
previousLoss = averageEpochLoss;

*/



        // Optional: Calculate and print test loss
        auto testOutput = neuralNetwork.forwardBatchNetwork(test_inputs);
        float testLoss = calculateLoss(testOutput.second, test_targets);
        std::cerr << "Test Loss at start of Epoch " << epoch << ": " << testLoss << std::endl;


        lossFile << "Epoch " << epoch << ", Loss: " << testLoss << std::endl;

// Check if this is the lowest RMSE so far
        if (testLoss < lowestRMSE) {
            lowestRMSE = testLoss;


        std::cerr << "Saving weights and biases for Test RMSE " << lowestRMSE <<  std::endl;



            // Save weights and biases
saveWeights(weights, "best_weights.txt");
            saveBiases(biases, "best_biases.txt");
        }


/*
size_t freeMemory, totalMemory;
cudaMemGetInfo(&freeMemory, &totalMemory);

std::cerr << "GPU Memory Usage at epoch end:" << std::endl;
std::cerr << "Free Memory: " << (freeMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cerr << "Total Memory: " << (totalMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cerr << "Used Memory: " << ((totalMemory - freeMemory) / 1024.0 / 1024.0) << " MB" << std::endl;
*/


 auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cerr << "Elapsed time: " << elapsed.count() << " s\n";







    }





return true;

}




private:
    int inputSize;
    int hiddenSize1;
    int hiddenSize2;
    int hiddenSize3;
    int hiddenSize4;
    int outputSize;
    float learningRate;
    std::vector<float> weightsHidden1;
    std::vector<float> weightsHidden2;
    std::vector<float> weightsHidden3;
    std::vector<float> weightsHidden4;
    std::vector<float> weightsOutput;
    std::vector<float> biasesHidden1;
    std::vector<float> biasesHidden2;
    std::vector<float> biasesHidden3;
    std::vector<float> biasesHidden4;
    std::vector<float> biasesOutput;
 std::vector<std::vector<float>> weights; // Member variable to store weights
    std::vector<std::vector<float>> biases;  // Member variable to store biases
std::vector<int> layerSizes;
    

    int numLayers;
    int batchSize;

    std::vector<std::vector<float>> layerErrors;
    std::vector<std::vector<float>> weightGradients;
    std::vector<std::vector<float>> biasGradients;
    std::vector<std::vector<float>> batchOutputs;
    std::vector<std::vector<float>> weightsTransposed;
    std::vector<float> transposedInput;




///////////////////////////


 // Forward pass
   std::vector<float> forward(const std::vector<float>& input) {
    // Debugging information
    std::cout << "Input: ";
    for (const auto& i : input) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // Check input size
    if (input.size() != inputSize) {
        std::cerr << "Input size mismatch." << std::endl;
        exit(1);
    }

    // Compute the hidden layer 1 output
    std::vector<float> hidden1(hiddenSize1, 0.0f);
    MultiplyMatricesCuBLAS(input, weightsHidden1, hidden1, 1, inputSize, hiddenSize1);
    for (int i = 0; i < hiddenSize1; ++i) {
        hidden1[i] += biasesHidden1[i];
        hidden1[i] = hidden1[i] > 0 ? hidden1[i] : 0.0f; // ReLU activation
    }

    // Debugging information for hidden layer 1
    std::cout << "Hidden Layer 1 Output: ";
    for (const auto& h : hidden1) {
        std::cout << h << " ";
    }
    std::cout << std::endl;

    // Compute the hidden layer 2 output
    std::vector<float> hidden2(hiddenSize2, 0.0f);
    MultiplyMatricesCuBLAS(hidden1, weightsHidden2, hidden2, 1, hiddenSize1, hiddenSize2);
    for (int i = 0; i < hiddenSize2; ++i) {
        hidden2[i] += biasesHidden2[i];
        hidden2[i] = hidden2[i] > 0 ? hidden2[i] : 0.0f; // ReLU activation
    }

    // Debugging information for hidden layer 2
    std::cout << "Hidden Layer 2 Output: ";
    for (const auto& h : hidden2) {
        std::cout << h << " ";
    }
    std::cout << std::endl;

    // Compute the hidden layer 3 output
    std::vector<float> hidden3(hiddenSize3, 0.0f);
    MultiplyMatricesCuBLAS(hidden2, weightsHidden3, hidden3, 1, hiddenSize2, hiddenSize3);
    for (int i = 0; i < hiddenSize3; ++i) {
        hidden3[i] += biasesHidden3[i];
        hidden3[i] = hidden3[i] > 0 ? hidden3[i] : 0.0f; // ReLU activation
    }

    // Debugging information for hidden layer 3
    std::cout << "Hidden Layer 3 Output: ";
    for (const auto& h : hidden3) {
        std::cout << h << " ";
    }
    std::cout << std::endl;

    // Compute the hidden layer 4 output
    std::vector<float> hidden4(hiddenSize4, 0.0f);
    MultiplyMatricesCuBLAS(hidden3, weightsHidden4, hidden4, 1, hiddenSize3, hiddenSize4);
    for (int i = 0; i < hiddenSize4; ++i) {
        hidden4[i] += biasesHidden4[i];
        hidden4[i] = hidden4[i] > 0 ? hidden4[i] : 0.0f; // ReLU activation
    }

    // Debugging information for hidden layer 4
    std::cout << "Hidden Layer 4 Output: ";
    for (const auto& h : hidden4) {
        std::cout << h << " ";
    }
    std::cout << std::endl;

    // Compute the output layer's output
    std::vector<float> output(outputSize, 0.0f);
    MultiplyMatricesCuBLAS(hidden4, weightsOutput, output, 1, hiddenSize4, outputSize);
    for (int i = 0; i < outputSize; ++i) {
        output[i] += biasesOutput[i];
    }

    // Debugging information for the output layer
    std::cout << "Output: ";
    for (const auto& o : output) {
        std::cout << o << " ";
    }
    std::cout << std::endl;

    return output;
}


/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////

// Helper function to check and report CUDA errors
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}





/////////////////////////////////
/////////////////////////////////
/////////////////////////////////











/////////////////////////////////
/////////////////////////////////
/////////////////////////////////








    // Backpropagation logic to update weights and biases
    void backpropagate(const std::vector<float>& input, const std::vector<float>& target) {
        // Check input and target sizes
        if (input.size() != inputSize || target.size() != outputSize) {
            std::cerr << "Input or target size mismatch." << std::endl;
            exit(1);
        }

        // Forward pass to compute activations
        std::vector<float> hidden1(hiddenSize1, 0.0f);
        MultiplyMatricesCuBLAS(input, weightsHidden1, hidden1, 1, inputSize, hiddenSize1);
        for (int i = 0; i < hiddenSize1; ++i) {
            hidden1[i] += biasesHidden1[i];
            hidden1[i] = hidden1[i] > 0 ? hidden1[i] : 0.0f; // ReLU activation
        }

        std::vector<float> hidden2(hiddenSize2, 0.0f);
        MultiplyMatricesCuBLAS(hidden1, weightsHidden2, hidden2, 1, hiddenSize1, hiddenSize2);
        for (int i = 0; i < hiddenSize2; ++i) {
            hidden2[i] += biasesHidden2[i];
            hidden2[i] = hidden2[i] > 0 ? hidden2[i] : 0.0f; // ReLU activation
        }

        std::vector<float> hidden3(hiddenSize3, 0.0f);
        MultiplyMatricesCuBLAS(hidden2, weightsHidden3, hidden3, 1, hiddenSize2, hiddenSize3);
        for (int i = 0; i < hiddenSize3; ++i) {
            hidden3[i] += biasesHidden3[i];
            hidden3[i] = hidden3[i] > 0 ? hidden3[i] : 0.0f; // ReLU activation
        }

        std::vector<float> hidden4(hiddenSize4, 0.0f);
        MultiplyMatricesCuBLAS(hidden3, weightsHidden4, hidden4, 1, hiddenSize3, hiddenSize4);
        for (int i = 0; i < hiddenSize4; ++i) {
            hidden4[i] += biasesHidden4[i];
            hidden4[i] = hidden4[i] > 0 ? hidden4[i] : 0.0f; // ReLU activation
        }

        std::vector<float> output(outputSize, 0.0f);
        MultiplyMatricesCuBLAS(hidden4, weightsOutput, output, 1, hiddenSize4, outputSize);
        for (int i = 0; i < outputSize; ++i) {
            output[i] += biasesOutput[i];
        }

        // Compute output layer error
        std::vector<float> outputError(outputSize, 0.0f);
        for (int i = 0; i < outputSize; ++i) {
            outputError[i] = output[i] - target[i];
        }

        // Backpropagate error through the layers
        std::vector<float> hiddenError4(hiddenSize4, 0.0f);
        MultiplyMatricesCuBLAS(outputError, weightsOutput, hiddenError4, 1, outputSize, hiddenSize4);
        applyDerivative(hiddenError4);

        std::vector<float> hiddenError3(hiddenSize3, 0.0f);
        MultiplyMatricesCuBLAS(hiddenError4, weightsHidden4, hiddenError3, 1, hiddenSize4, hiddenSize3);
        applyDerivative(hiddenError3);

        std::vector<float> hiddenError2(hiddenSize2, 0.0f);
        MultiplyMatricesCuBLAS(hiddenError3, weightsHidden3, hiddenError2, 1, hiddenSize3, hiddenSize2);
        applyDerivative(hiddenError2);

        std::vector<float> hiddenError1(hiddenSize1, 0.0f);
        MultiplyMatricesCuBLAS(hiddenError2, weightsHidden2, hiddenError1, 1, hiddenSize2, hiddenSize1);
        applyDerivative(hiddenError1);

        // Update weights and biases
        updateWeightsAndBiases(input, hidden1, hidden2, hidden3, hidden4, outputError, hiddenError4, hiddenError3, hiddenError2, hiddenError1);
    }


/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////

   
};

// Additional functions for RMSE calculation, training process, etc.
float calculateRMSE(const std::vector<float>& outputs, const std::vector<float>& targets) {
    float mse = 0.0f;
    for (size_t i = 0; i < outputs.size(); ++i) {
        float diff = outputs[i] - targets[i];
        mse += diff * diff;
    }
    return std::sqrt(mse / outputs.size());
}




/*
int main() {

 // Start timing
    auto start = std::chrono::high_resolution_clock::now();

size_t freeMemory, totalMemory;
cudaMemGetInfo(&freeMemory, &totalMemory);

std::cout << "GPU Memory Usage:" << std::endl;
std::cout << "Free Memory: " << (freeMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Total Memory: " << (totalMemory / 1024.0 / 1024.0) << " MB" << std::endl;
std::cout << "Used Memory: " << ((totalMemory - freeMemory) / 1024.0 / 1024.0) << " MB" << std::endl;


   const int NUM_FEATURES = 14;
    const int NUM_TRAIN_SAMPLES = 800;
    const int NUM_TEST_SAMPLES = 20000;

    std::vector<std::vector<float>> train_inputs(NUM_TRAIN_SAMPLES, std::vector<float>(NUM_FEATURES));
    std::vector<std::vector<float>> train_targets(NUM_TRAIN_SAMPLES, std::vector<float>(1)); // Single output per sample
    std::vector<std::vector<float>> test_inputs(NUM_TEST_SAMPLES, std::vector<float>(NUM_FEATURES));
    std::vector<std::vector<float>> test_targets(NUM_TEST_SAMPLES, std::vector<float>(1));

    // Load data from a file
    std::ifstream file("networkInput.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    std::string line;
    for (int i = 0; i < NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES; ++i) {
        if (!getline(file, line)) break;
        std::istringstream iss(line);
        if (i < NUM_TRAIN_SAMPLES) {
            for (int j = 0; j < NUM_FEATURES; ++j) {
                iss >> train_inputs[i][j];
            }
            iss >> train_targets[i][0];
        } else {
            for (int j = 0; j < NUM_FEATURES; ++j) {
                iss >> test_inputs[i - NUM_TRAIN_SAMPLES][j];
            }
            iss >> test_targets[i - NUM_TRAIN_SAMPLES][0];
        }
    }
    file.close();

    // Define neural network architecture and hyperparameters
    const int inputSize = NUM_FEATURES;
    const int hiddenSize1 = 30;
    const int hiddenSize2 = 15;
    const int hiddenSize3 = 7;
    const int hiddenSize4 = 3;
    const int outputSize = 1;
    const float learningRate = 1.0f;
    const int numEpochs = 200;


    // Initialize the neural network
    NeuralNetwork neuralNetwork(inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize, learningRate);

    // Train the neural network
    neuralNetwork.trainNeuralNetwork(neuralNetwork, train_inputs, train_targets,test_inputs, test_targets, learningRate, numEpochs);

    // Test the network again after training (optional)
   // auto testOutput = neuralNetwork.forwardBatchNetwork(test_inputs);

    // Print the outputs after backpropagation
    std::cout << "Outputs after Backpropagation:" << std::endl;



// End timing
    auto finish = std::chrono::high_resolution_clock::now();
    
    // Calculate elapsed time
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";


    return 0;

}
*/


int main() {
    // Start timing
    //auto start = std::chrono::high_resolution_clock::now();

    const int NUM_FEATURES = 14;
    
    // Neural network setup
    const int inputSize = NUM_FEATURES;
    const int hiddenSize1 = 30;
    const int hiddenSize2 = 15;
    const int hiddenSize3 = 7;
    const int hiddenSize4 = 3;
    const int outputSize = 1;

     float learningRate = 0.00025f;
    const int numEpochs = 200;

    const int NUM_TRAIN_SAMPLES = 20000;
    const int NUM_TEST_SAMPLES = 5000;
    int BATCH_SIZE = 20000;

    

    
/*
    NeuralNetwork neuralNetwork(inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize, learningRate,BATCH_SIZE);
*/
    // Train the neural network with batches
/*
    neuralNetwork.trainNeuralNetwork(neuralNetwork, train_inputs_batches, train_targets_batches, test_inputs, test_targets, learningRate, numEpochs);
*/


// Limit for training attempts
   const int maxAttempts = 10;
bool trainingSuccessful = false;

for (int attempt = 0; attempt < maxAttempts && !trainingSuccessful; ++attempt) {
    std::cout << "Training attempt " << (attempt + 1) << std::endl;


learningRate*=0.8f;



std::vector<std::vector<float>> train_inputs(NUM_TRAIN_SAMPLES, std::vector<float>(NUM_FEATURES));
    std::vector<std::vector<float>> train_targets(NUM_TRAIN_SAMPLES, std::vector<float>(1));
    std::vector<std::vector<float>> test_inputs(NUM_TEST_SAMPLES, std::vector<float>(NUM_FEATURES));
    std::vector<std::vector<float>> test_targets(NUM_TEST_SAMPLES, std::vector<float>(1));
    // Load data from a file
    std::ifstream file("networkInput.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    std::string line;
    for (int i = 0; i < NUM_TRAIN_SAMPLES + NUM_TEST_SAMPLES; ++i) {
        if (!getline(file, line)) break;
        std::istringstream iss(line);
        if (i < NUM_TRAIN_SAMPLES) {
            for (int j = 0; j < NUM_FEATURES; ++j) {
                iss >> train_inputs[i][j];
            }
            iss >> train_targets[i][0];
        } else {
            for (int j = 0; j < NUM_FEATURES; ++j) {
                iss >> test_inputs[i - NUM_TRAIN_SAMPLES][j];
            }
            iss >> test_targets[i - NUM_TRAIN_SAMPLES][0];
        }
    }
    file.close();

    // Splitting training data into batches
    std::vector<std::vector<std::vector<float>>> train_inputs_batches;
    std::vector<std::vector<std::vector<float>>> train_targets_batches;

    for (int i = 0; i < NUM_TRAIN_SAMPLES; i += BATCH_SIZE) {
        std::vector<std::vector<float>> input_batch(train_inputs.begin() + i, train_inputs.begin() + std::min(i + BATCH_SIZE, NUM_TRAIN_SAMPLES));
        std::vector<std::vector<float>> target_batch(train_targets.begin() + i, train_targets.begin() + std::min(i + BATCH_SIZE, NUM_TRAIN_SAMPLES));

        train_inputs_batches.push_back(input_batch);
        train_targets_batches.push_back(target_batch);
    }










    // Reinitialize the neural network before each attempt to ensure a fresh start
    NeuralNetwork neuralNetwork(inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize, learningRate, BATCH_SIZE);
    // Attempt to train the neural network
    trainingSuccessful = neuralNetwork.trainNeuralNetwork(neuralNetwork, train_inputs_batches, train_targets_batches, test_inputs, test_targets, learningRate, numEpochs);

    if (!trainingSuccessful) {
        std::cerr << "Training was unsuccessful, restarting..." << std::endl;
        // No need for a second call to trainNeuralNetwork here, as the loop will continue to the next iteration
    }
}

if (!trainingSuccessful) {
    std::cerr << "Training failed after " << maxAttempts << " attempts." << std::endl;
} else {
    std::cout << "Training completed successfully." << std::endl;
    // Optionally, test the network again after training and print outputs
}



    // Optionally, test the network again after training and print outputs

    // End timing
    //auto finish = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed = finish - start;
    //std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    return 0;
}
