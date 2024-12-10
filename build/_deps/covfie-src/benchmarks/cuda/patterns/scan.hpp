/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <chrono>
#include <iostream>
#include <random>

#include <covfie/benchmark/pattern.hpp>
#include <covfie/benchmark/propagate.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/cuda/error_check.hpp>

template <typename backend_t>
__global__ void
scan_kernel(float * out, covfie::field_view<backend_t> f, int x, int y, int z)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x +
                  gridDim.x * gridDim.y * blockIdx.z;

    int threadId = blockId * blockDim.x + threadIdx.x;

    if (tx >= x || ty >= y || tz >= z) {
        return;
    }

    typename std::decay_t<decltype(f)>::output_t b = f.at(
        -10000.f + tx * 100.f, -10000.f + ty * 100.f, -15000.f + tz * 100.f
    );

    out[threadId] = b[0] + b[1] + b[2];
}

struct Scan : covfie::benchmark::AccessPattern<Scan> {
    struct parameters {
        std::size_t x, y, z;
    };

    static constexpr std::array<std::string_view, 3> parameter_names = {
        "X", "Y", "Z"};

    template <typename backend_t>
    static void iteration(
        const parameters & p,
        const covfie::field_view<backend_t> & f,
        ::benchmark::State & state
    )
    {
        state.PauseTiming();

        float * device_out = nullptr;

        cudaErrorCheck(cudaMalloc(&device_out, p.x * p.y * p.z * sizeof(float))
        );

        std::chrono::high_resolution_clock::time_point begin =
            std::chrono::high_resolution_clock::now();

        state.ResumeTiming();

        dim3 block_size(8u, 8u, 8u);
        dim3 grid_size(
            static_cast<unsigned int>(
                p.x / block_size.x + (p.x % block_size.x != 0 ? 1 : 0)
            ),
            static_cast<unsigned int>(
                p.y / block_size.y + (p.y % block_size.y != 0 ? 1 : 0)
            ),
            static_cast<unsigned int>(
                p.z / block_size.z + (p.z % block_size.z != 0 ? 1 : 0)
            )
        );

        scan_kernel<<<grid_size, block_size>>>(
            device_out,
            f,
            static_cast<int>(p.x),
            static_cast<int>(p.y),
            static_cast<int>(p.z)
        );

        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());

        state.PauseTiming();

        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();

        cudaErrorCheck(cudaFree(device_out));

        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end - begin
            );
    }

    static std::vector<std::vector<int64_t>> get_parameter_ranges()
    {
        return {{200}, {200}, {300}};
    }

    static parameters get_parameters(benchmark::State & state)
    {
        return {
            static_cast<std::size_t>(state.range(0)),
            static_cast<std::size_t>(state.range(1)),
            static_cast<std::size_t>(state.range(2))};
    }

    template <typename S, std::size_t N>
    static covfie::benchmark::counters get_counters(const parameters & p)
    {
        return {p.x * p.y * p.z, p.x * p.y * p.z * N * sizeof(S), 0};
    }
};
