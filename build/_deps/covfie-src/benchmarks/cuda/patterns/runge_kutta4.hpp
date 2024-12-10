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
__global__ void rk4_kernel(
    covfie::benchmark::propagation_agent<3> * agents,
    std::size_t n_agents,
    std::size_t n_steps,
    covfie::field_view<backend_t> f
)
{
    static constexpr float ss = 0.001f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_agents) {
        return;
    }

    covfie::benchmark::propagation_agent<3> o = agents[tid];

    for (std::size_t i = 0; i < n_steps; ++i) {
        propagation_step<RungeKutta4>(f, o, ss);
    }

    agents[tid] = o;
}

struct RungeKutta4Pattern
    : covfie::benchmark::AccessPattern<RungeKutta4Pattern> {
    struct parameters {
        std::size_t agents, block_size, steps;
    };

    static constexpr std::array<std::string_view, 3> parameter_names = {
        "N", "B", "S"};

    template <typename backend_t>
    static void iteration(
        const parameters & p,
        const covfie::field_view<backend_t> & f,
        ::benchmark::State & state
    )
    {
        state.PauseTiming();

        std::random_device rd;
        std::mt19937 e(rd());

        std::uniform_real_distribution<float> pos_dist(-1000.0, 1000.0);

        std::vector<covfie::benchmark::propagation_agent<3>> objs(p.agents);

        for (std::size_t i = 0; i < objs.size(); ++i) {
            objs[i].pos[0] = pos_dist(e);
            objs[i].pos[1] = pos_dist(e);
            objs[i].pos[2] = pos_dist(e);
        }

        covfie::benchmark::propagation_agent<3> * device_agents = nullptr;

        cudaErrorCheck(cudaMalloc(
            &device_agents,
            p.agents * sizeof(covfie::benchmark::propagation_agent<3>)
        ));
        cudaErrorCheck(cudaMemcpy(
            device_agents,
            objs.data(),
            p.agents * sizeof(covfie::benchmark::propagation_agent<3>),
            cudaMemcpyHostToDevice
        ));

        std::chrono::high_resolution_clock::time_point begin =
            std::chrono::high_resolution_clock::now();

        state.ResumeTiming();

        rk4_kernel<<<
            static_cast<unsigned int>(
                p.agents / p.block_size + (p.agents % p.block_size != 0 ? 1 : 0)
            ),
            static_cast<unsigned int>(p.block_size)>>>(
            device_agents, p.agents, p.steps, f
        );

        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());

        state.PauseTiming();

        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();

        cudaErrorCheck(cudaFree(device_agents));

        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end - begin
            );
    }

    static std::vector<std::vector<int64_t>> get_parameter_ranges()
    {
        return {
            {4096, 16384, 65536, 262144, 1048576, 4194304},
            {32, 64, 128, 256, 512},
            {1024, 2048, 4096, 8192, 16384, 32768, 65536}};
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
        return {
            p.agents * p.steps,
            p.agents * p.steps * N * sizeof(S),
            p.agents * p.steps * 21};
    }
};
