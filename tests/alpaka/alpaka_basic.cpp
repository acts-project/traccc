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
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

#include "traccc/alpaka/utils/vecmem_types.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED
#include <sycl/sycl.hpp>
#include <vecmem/utils/sycl/queue_wrapper.hpp>
#endif

// GoogleTest include(s).
#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>

template <typename Acc>
ALPAKA_FN_ACC float process(Acc const& acc, uint32_t idx) {
    return static_cast<float>(alpaka::math::sin(acc, idx));
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
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = getDevByIdx(platformAcc, 0u);

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
    auto const platformDevCpu = alpaka::Platform<DevCpu>{};
    auto devHost = getDevByIdx(platformDevCpu, 0u);

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
        EXPECT_FLOAT_EQ(bufHost[i], static_cast<float>(std::sin(i)));
    }
}

struct VecMemOpKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc, vecmem::data::vector_view<float> result) const {
        using namespace alpaka;
        auto const globalThreadIdx =
            getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        // It is possible to get index type from the accelerator,
        // but for simplicity we just repeat the type here
        if (globalThreadIdx < result.size()) {
            result.ptr()[globalThreadIdx] = process(acc, globalThreadIdx);
        }
    }
};

GTEST_TEST(AlpakaBasic, VecMemOp) {

    using namespace alpaka;
    using Dim = DimInt<1>;
    using Idx = uint32_t;

    // Select a device and create queue for it
    using Acc = ExampleDefaultAcc<Dim, Idx>;
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = getDevByIdx(platformAcc, 0u);

    using Queue = Queue<Acc, Blocking>;
    auto queue = Queue{devAcc};

    uint32_t n = 10000;

    uint32_t blocksPerGrid = n;
    uint32_t threadsPerBlock = 1;
    uint32_t elementsPerThread = 4;
    using WorkDiv = WorkDivMembers<Dim, Idx>;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};

#ifdef ALPAKA_ACC_SYCL_ENABLED
    ::sycl::queue q;
    vecmem::sycl::queue_wrapper qw{&q};
    traccc::alpaka::vecmem::device_copy vm_copy(qw);
#else
    traccc::alpaka::vecmem::device_copy vm_copy;
#endif

    traccc::alpaka::vecmem::host_memory_resource host_mr;
    traccc::alpaka::vecmem::device_memory_resource device_mr;

    vecmem::vector<float> host_vector{n, &host_mr};

    auto host_buffer = vecmem::get_data(host_vector);
    auto device_buffer = vm_copy.to(vecmem::get_data(host_vector), device_mr,
                                    vecmem::copy::type::host_to_device);
    auto data_dev_vec_buf = vecmem::get_data(device_buffer);

    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>()
              << std::endl;

    // Create a device for host for memory allocation, using the first CPU
    // available
    auto const platformDevCpu = alpaka::Platform<DevCpu>{};
    auto devHost = getDevByIdx(platformDevCpu, 0u);

    alpaka::exec<Acc>(queue, workDiv, VecMemOpKernel{}, data_dev_vec_buf);

    vm_copy(device_buffer, host_buffer, vecmem::copy::type::device_to_host)
        ->wait();

    for (uint32_t i = 0u; i < n; i++) {
        EXPECT_FLOAT_EQ(host_vector[i], static_cast<float>(std::sin(i)));
    }
}
