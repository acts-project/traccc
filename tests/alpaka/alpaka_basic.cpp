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

struct VectorOpKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, float* result,
                                  uint32_t n) const {
        using namespace alpaka;
        auto const globalThreadIdx =
            getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        // It is possible to get index type from the accelerator,
        // but for simplicity we just repeat the type here
        if (globalThreadIdx < n) {
            result[globalThreadIdx] = process(acc, globalThreadIdx);
        }
    }
};

/// Copy vector to device, set element[i] = sin(i) on device, then compare with
/// same calculation on host
GTEST_TEST(AlpakaBasic, VectorOp) {
    using namespace alpaka;
    using Dim = DimInt<1>;
    using Idx = uint32_t;

    using Acc = ExampleDefaultAcc<Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>()
              << std::endl;

    // Select a device and create queue for it
    auto const devAcc = getDevByIdx<Acc>(0u);

    using Queue = Queue<Acc, Blocking>;
    auto queue = Queue{devAcc};

    uint32_t n = 10000;

    uint32_t blocksPerGrid = n;
    uint32_t threadsPerBlock = 1;
    uint32_t elementsPerThread = 4;
    using WorkDiv = WorkDivMembers<Dim, Idx>;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};

    // Create a device for host for memory allocation, using the first CPU
    // available
    auto devHost = getDevByIdx<DevCpu>(0u);

    // Allocate memory on the device side:
    auto bufAcc = alpaka::allocBuf<float, uint32_t>(devAcc, n);

    alpaka::exec<Acc>(queue, workDiv, VectorOpKernel{},
                      alpaka::getPtrNative(bufAcc), n);

    alpaka::wait(queue);

    // Allocate memory on the host side:
    auto bufHost = alpaka::allocBuf<float, uint32_t>(devHost, n);
    // Copy bufAcc to bufHost
    alpaka::memcpy(queue, bufHost, bufAcc);
    // Calculate on the host and compare result
    for (uint32_t i = 0u; i < n; i++) {
        EXPECT_FLOAT_EQ(bufHost[i], std::sin(i));
    }
}
