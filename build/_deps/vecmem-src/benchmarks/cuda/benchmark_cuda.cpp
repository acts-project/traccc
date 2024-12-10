/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

#include <vector>

static vecmem::cuda::device_memory_resource device_mr;
void BenchmarkCudaDevice(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        void* p = device_mr.allocate(size);
        device_mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkCudaDevice)->RangeMultiplier(2)->Range(1, 2UL << 31);

void BenchmarkCudaDeviceBinaryPage(benchmark::State& state) {
    std::size_t size = static_cast<std::size_t>(state.range(0));

    vecmem::binary_page_memory_resource mr(device_mr);

    for (auto _ : state) {
        void* p = mr.allocate(size);
        mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkCudaDeviceBinaryPage)
    ->RangeMultiplier(2)
    ->Range(1, 2UL << 31);

void BenchmarkCudaDeviceBinaryPageMultiple(benchmark::State& state) {
    std::size_t size = static_cast<std::size_t>(state.range(0));
    std::size_t nallocs = static_cast<std::size_t>(state.range(1));

    std::vector<void*> allocs;

    allocs.reserve(nallocs);

    for (auto _ : state) {
        vecmem::binary_page_memory_resource mr(device_mr);
        for (std::size_t i = 0; i < nallocs; ++i) {
            allocs[i] = mr.allocate(size);
        }

        for (std::size_t i = 0; i < nallocs; ++i) {
            mr.deallocate(allocs[i], size);
        }
    }
}
BENCHMARK(BenchmarkCudaDeviceBinaryPageMultiple)
    ->RangeMultiplier(2)
    ->Ranges({{1, 2UL << 21}, {1, 1024}});

static vecmem::cuda::host_memory_resource host_mr;
void BenchmarkCudaPinned(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        void* p = host_mr.allocate(size);
        host_mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkCudaPinned)->RangeMultiplier(2)->Range(1, 2UL << 31);

void BenchmarkCudaPinnedBinaryPage(benchmark::State& state) {
    std::size_t size = static_cast<std::size_t>(state.range(0));

    vecmem::binary_page_memory_resource mr(host_mr);

    for (auto _ : state) {
        void* p = mr.allocate(size);
        mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkCudaPinnedBinaryPage)
    ->RangeMultiplier(2)
    ->Range(1, 2UL << 31);

static vecmem::cuda::managed_memory_resource managed_mr;
void BenchmarkCudaManaged(benchmark::State& state) {
    const std::size_t size = static_cast<std::size_t>(state.range(0));
    for (auto _ : state) {
        void* p = managed_mr.allocate(size);
        managed_mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkCudaManaged)->RangeMultiplier(2)->Range(1, 2UL << 31);

void BenchmarkCudaManagedBinaryPage(benchmark::State& state) {
    std::size_t size = static_cast<std::size_t>(state.range(0));

    vecmem::binary_page_memory_resource mr(managed_mr);

    for (auto _ : state) {
        void* p = mr.allocate(size);
        mr.deallocate(p, size);
    }
}
BENCHMARK(BenchmarkCudaManagedBinaryPage)
    ->RangeMultiplier(2)
    ->Range(1, 2UL << 31);
