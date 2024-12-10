/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/utils/sycl/copy.hpp>
#include <vecmem/utils/sycl/queue_wrapper.hpp>

// Common benchmark include(s).
#include "../common/make_jagged_sizes.hpp"
#include "../common/make_jagged_vector.hpp"

// Google benchmark include(s).
#include <benchmark/benchmark.h>

// System include(s).
#include <numeric>
#include <vector>

namespace vecmem::sycl::benchmark {

/// The queue that the VecMem objects should use
static queue_wrapper queue;
/// The (host) memory resource to use in the benchmark(s).
static host_memory_resource host_mr{queue};
/// The (device) memory resource to use in the benchmark(s).
static device_memory_resource device_mr{queue};
/// The copy object to use in the benchmark(s).
static copy sycl_copy{queue};

/// Function benchmarking "unknown" host-to-device jagged vector copies
void jaggedVectorUnknownHtoDCopy(::benchmark::State& state) {

    // Generate the sizes of the jagged vector/buffer for the test.
    const std::vector<std::size_t> sizes =
        vecmem::benchmark::make_jagged_sizes(state.range(0), state.range(1));

    // Set custom "counters" for the benchmark.
    const std::size_t bytes = std::accumulate(sizes.begin(), sizes.end(),
                                              static_cast<std::size_t>(0u)) *
                              sizeof(int);
    state.counters["Bytes"] = static_cast<double>(bytes);
    state.counters["Rate"] =
        ::benchmark::Counter(static_cast<double>(bytes),
                             ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::kIs1024);

    // Create the "source vector".
    jagged_vector<int> source =
        vecmem::benchmark::make_jagged_vector(sizes, host_mr);
    const data::jagged_vector_data<int> source_data = get_data(source);
    // Create the "destination buffer".
    data::jagged_vector_buffer<int> dest(sizes, device_mr, &host_mr);
    sycl_copy.setup(dest)->wait();

    // Perform the copy benchmark.
    for (auto _ : state) {
        sycl_copy(source_data, dest)->wait();
    }
}
// Set up the benchmark.
BENCHMARK(jaggedVectorUnknownHtoDCopy)->Ranges({{10, 100000}, {50, 5000}});

/// Function benchmarking "known" host-to-device jagged vector copies
void jaggedVectorKnownHtoDCopy(::benchmark::State& state) {

    // Generate the sizes of the jagged vector/buffer for the test.
    const std::vector<std::size_t> sizes =
        vecmem::benchmark::make_jagged_sizes(state.range(0), state.range(1));

    // Set custom "counters" for the benchmark.
    const std::size_t bytes = std::accumulate(sizes.begin(), sizes.end(),
                                              static_cast<std::size_t>(0u)) *
                              sizeof(int);
    state.counters["Bytes"] = static_cast<double>(bytes);
    state.counters["Rate"] =
        ::benchmark::Counter(static_cast<double>(bytes),
                             ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::kIs1024);

    // Create the "source vector".
    jagged_vector<int> source =
        vecmem::benchmark::make_jagged_vector(sizes, host_mr);
    const data::jagged_vector_data<int> source_data = get_data(source);
    // Create the "destination buffer".
    data::jagged_vector_buffer<int> dest(sizes, device_mr, &host_mr);
    sycl_copy.setup(dest)->wait();

    // Perform the copy benchmark.
    for (auto _ : state) {
        sycl_copy(source_data, dest, copy::type::host_to_device)->wait();
    }
}
// Set up the benchmark.
BENCHMARK(jaggedVectorKnownHtoDCopy)->Ranges({{10, 100000}, {50, 5000}});

/// Function benchmarking "unknown" device-to-host jagged vector copies
void jaggedVectorUnknownDtoHCopy(::benchmark::State& state) {

    // Generate the sizes of the jagged vector/buffer for the test.
    const std::vector<std::size_t> sizes =
        vecmem::benchmark::make_jagged_sizes(state.range(0), state.range(1));

    // Set custom "counters" for the benchmark.
    const std::size_t bytes = std::accumulate(sizes.begin(), sizes.end(),
                                              static_cast<std::size_t>(0u)) *
                              sizeof(int);
    state.counters["Bytes"] = static_cast<double>(bytes);
    state.counters["Rate"] =
        ::benchmark::Counter(static_cast<double>(bytes),
                             ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::kIs1024);

    // Create the "source buffer".
    data::jagged_vector_buffer<int> source(sizes, device_mr, &host_mr);
    sycl_copy.setup(source)->wait();
    // Create the "destination vector".
    jagged_vector<int> dest =
        vecmem::benchmark::make_jagged_vector(sizes, host_mr);
    data::jagged_vector_data<int> dest_data = get_data(dest);

    // Perform the copy benchmark.
    for (auto _ : state) {
        sycl_copy(source, dest_data)->wait();
    }
}
// Set up the benchmark.
BENCHMARK(jaggedVectorUnknownDtoHCopy)->Ranges({{10, 100000}, {50, 5000}});

/// Function benchmarking "known" device-to-host jagged vector copies
void jaggedVectorKnownDtoHCopy(::benchmark::State& state) {

    // Generate the sizes of the jagged vector/buffer for the test.
    const std::vector<std::size_t> sizes =
        vecmem::benchmark::make_jagged_sizes(state.range(0), state.range(1));

    // Set custom "counters" for the benchmark.
    const std::size_t bytes = std::accumulate(sizes.begin(), sizes.end(),
                                              static_cast<std::size_t>(0u)) *
                              sizeof(int);
    state.counters["Bytes"] = static_cast<double>(bytes);
    state.counters["Rate"] =
        ::benchmark::Counter(static_cast<double>(bytes),
                             ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::kIs1024);

    // Create the "source buffer".
    data::jagged_vector_buffer<int> source(sizes, device_mr, &host_mr);
    sycl_copy.setup(source)->wait();
    // Create the "destination vector".
    jagged_vector<int> dest =
        vecmem::benchmark::make_jagged_vector(sizes, host_mr);
    data::jagged_vector_data<int> dest_data = get_data(dest);

    // Perform the copy benchmark.
    for (auto _ : state) {
        sycl_copy(source, dest_data, copy::type::device_to_host)->wait();
    }
}
// Set up the benchmark.
BENCHMARK(jaggedVectorKnownDtoHCopy)->Ranges({{10, 100000}, {50, 5000}});

}  // namespace vecmem::sycl::benchmark
