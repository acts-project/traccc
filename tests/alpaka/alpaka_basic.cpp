/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Alpaka include(s).
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

// GoogleTest include(s).
#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>

template <typename Acc>
ALPAKA_FN_ACC float process(Acc const& acc, uint32_t idx) {
    return alpaka::math::sin(acc, idx);
}

// Test if the given buffer has a correct result.
template<typename TQueue, typename TBufAcc>
void testResult(TQueue& queue, TBufAcc& bufAcc)
{
    // Wait for kernel to finish
    alpaka::wait(queue);
    // Copy results to host
    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const devHost = alpaka::getDevByIdx<alpaka::DevCpu>(0u);
    auto bufHost = alpaka::allocBuf<float, uint32_t>(devHost, n);
    alpaka::memcpy(queue, bufHost, bufAcc);
    // Reset values of device buffer
    auto const byte(static_cast<uint8_t>(0u));
    alpaka::memset(queue, bufAcc, byte);
    // Test that all elements were processed
    auto const* result = alpaka::getPtrNative(bufHost);
    bool testPassed = true;
    for(uint32_t i = 0u; i < n; i++)
        testPassed = testPassed && (std::abs(result[i] - process(devHost, i)) < 1e-3);
    std::cout << (testPassed ? "Test passed.\n" : "Test failed.\n");
}

// Helper type to set alpaka kernel launch configuration
using WorkDiv = alpaka::WorkDivMembers<alpaka::DimInt<1u>, uint32_t>;

struct NaiveCudaStyleKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, float* result,
                                  uint32_t n) const {
        using namespace alpaka;
        auto const globalThreadIdx =
            getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        if (globalThreadIdx < n) {
            result[globalThreadIdx] = process(acc, globalThreadIdx);
        }
    }
};

template<typename Acc, typename Queue, typename BufAcc>
void naiveCudaStyle(Queue& queue, BufAcc& bufAcc) {

    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const deviceProperties = alpaka::getAccDevProps<Acc>(alpaka::getDevByIdx<Acc>(0u));
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    // Fixed number of threads per block.
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    auto const elementsPerThread = 1u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "\nNaive CUDA style processing - each thread processes one data point:\n";
    std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
              << "alpaka element layer not used\n";
    alpaka::exec<Acc>(queue, workDiv, NaiveCudaStyleKernel{}, alpaka::getPtrNative(bufAcc), n);
    testResult(queue, bufAcc);
}

struct GridStridedLoopKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, float* result,
                                  uint32_t n) const {
        auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

        for(uint32_t dataDomainIdx = globalThreadIdx; dataDomainIdx < n; dataDomainIdx += globalThreadExtent)
        {
            auto const memoryIdx = dataDomainIdx;
            result[memoryIdx] = process(acc, dataDomainIdx);
        }
    }
};

template<typename Acc, typename Queue, typename BufAcc>
void gridStridedLoop(Queue& queue, BufAcc& bufAcc) {

    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const deviceProperties = alpaka::getAccDevProps<Acc>(alpaka::getDevByIdx<Acc>(0u));
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    // Fixed number of threads per block.
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid = deviceProperties.m_multiProcessorCount;
    auto const elementsPerThread = 1u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "\nGrid strided loop processing - fixed number of threads and blocks:\n";
    std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
              << "alpaka element layer not used\n";
    alpaka::exec<Acc>(queue, workDiv, GridStridedLoopKernel{}, alpaka::getPtrNative(bufAcc), n);
    testResult(queue, bufAcc);
}

struct ChunkedGridStridedLoopKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, float* result,
                                  uint32_t n) const {
        auto const numElements(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        // Additionally could split the loop into peeled and remainder
        for(uint32_t chunkStart = globalThreadIdx * numElements; chunkStart < n;
            chunkStart += globalThreadExtent * numElements)
        {
            // When applicable, this loop can be done in vector fashion
            for(uint32_t dataDomainIdx = chunkStart; (dataDomainIdx < chunkStart + numElements) && (dataDomainIdx < n);
                dataDomainIdx++)
            {
                auto const memoryIdx = dataDomainIdx;
                result[memoryIdx] = process(acc, dataDomainIdx);
            }
        }
    }
};

template<typename Acc, typename Queue, typename BufAcc>
void chunkedGridStridedLoop(Queue& queue, BufAcc& bufAcc) {

    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const deviceProperties = alpaka::getAccDevProps<Acc>(alpaka::getDevByIdx<Acc>(0u));
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    // Fixed number of threads per block.
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid = deviceProperties.m_multiProcessorCount;
    auto const elementsPerThread = 8u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "\nChunked grid strided loop processing - fixed number of threads and blocks:\n";
    std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
              << elementsPerThread << " alpaka elements per thread\n";
    alpaka::exec<Acc>(queue, workDiv, ChunkedGridStridedLoopKernel{}, alpaka::getPtrNative(bufAcc), n);
    testResult(queue, bufAcc);
}

/// Copy vector to device, set element[i] = sin(i) on device, then compare with
/// same calculation on host
GTEST_TEST(AlpakaBasic, VectorOp) {
    using namespace alpaka;
    using Dim = DimInt<1>;
    using Idx = uint32_t;

    using Acc = ExampleDefaultAcc<Dim, Idx>;
    using Queue = Queue<Acc, Blocking>;

    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>()
              << std::endl;

    // Select a device and create queue for it
    auto const devAcc = getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};

    // Allocate memory on the device side:
    uint32_t n = 1 << 12;
    auto bufAcc = alpaka::allocBuf<float, uint32_t>(devAcc, n);

    // Run the kernel, test its result at the end.
    naiveCudaStyle<Acc>(queue, bufAcc);
    gridStridedLoop<Acc>(queue, bufAcc);
    chunkedGridStridedLoop<Acc>(queue, bufAcc);
}

