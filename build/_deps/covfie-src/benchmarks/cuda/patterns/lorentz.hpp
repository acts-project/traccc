/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <chrono>
#include <iostream>
#include <random>

#include <covfie/benchmark/lorentz.hpp>
#include <covfie/benchmark/pattern.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/cuda/error_check.hpp>

template <typename propagator_t, typename backend_t>
__global__ void lorentz_kernel(
    covfie::benchmark::lorentz_agent<3> * agents,
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

    covfie::benchmark::lorentz_agent<3> o = agents[tid];

    for (std::size_t i = 0; i < n_steps; ++i) {
        lorentz_step<propagator_t>(f, o, ss);
    }

    agents[tid] = o;
}

template <typename Propagator>
struct Lorentz : covfie::benchmark::AccessPattern<Lorentz<Propagator>> {
    struct parameters {
        std::size_t particles, block_size, steps, imom;
    };

    static constexpr std::array<std::string_view, 4> parameter_names = {
        "N", "B", "S", "I"};

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

        std::uniform_real_distribution<> phi_dist(0.f, 2.f * 3.1415927f);
        std::uniform_real_distribution<> costheta_dist(-1.f, 1.f);

        std::vector<covfie::benchmark::lorentz_agent<3>> objs(p.particles);

        for (std::size_t i = 0; i < p.particles; ++i) {
            float theta = std::acos(static_cast<float>(costheta_dist(e)));
            float phi = static_cast<float>(phi_dist(e));

            objs[i].pos[0] = 0.f;
            objs[i].pos[1] = 0.f;
            objs[i].pos[2] = 0.f;

            objs[i].mom[0] =
                static_cast<float>(p.imom) * std::sin(theta) * std::cos(phi);
            objs[i].mom[1] =
                static_cast<float>(p.imom) * std::sin(theta) * std::sin(phi);
            objs[i].mom[2] = static_cast<float>(p.imom) * std::cos(theta);
        }

        covfie::benchmark::lorentz_agent<3> * device_agents = nullptr;

        cudaErrorCheck(cudaMalloc(
            &device_agents,
            p.particles * sizeof(covfie::benchmark::lorentz_agent<3>)
        ));
        cudaErrorCheck(cudaMemcpy(
            device_agents,
            objs.data(),
            p.particles * sizeof(covfie::benchmark::lorentz_agent<3>),
            cudaMemcpyHostToDevice
        ));

        std::chrono::high_resolution_clock::time_point begin =
            std::chrono::high_resolution_clock::now();

        state.ResumeTiming();

        lorentz_kernel<Propagator>
            <<<static_cast<unsigned int>(
                   p.particles / p.block_size +
                   (p.particles % p.block_size != 0 ? 1 : 0)
               ),
               static_cast<unsigned int>(p.block_size)>>>(
                device_agents, p.particles, p.steps, f
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
            {1024, 2048, 4096, 8192, 16384, 32768, 65536},
            {128,   192,   256,    384,    512,    768,    1024,  1536,  2048,
             3072,  4096,  6144,   8192,   12288,  16384,  24576, 32768, 49152,
             65536, 98304, 131072, 196608, 262144, 393216, 524288}};
    }

    static parameters get_parameters(benchmark::State & state)
    {
        return {
            static_cast<std::size_t>(state.range(0)),
            static_cast<std::size_t>(state.range(1)),
            static_cast<std::size_t>(state.range(2)),
            static_cast<std::size_t>(state.range(3))};
    }

    template <typename S, std::size_t N>
    static covfie::benchmark::counters get_counters(const parameters & p)
    {
        return {
            p.particles * p.steps,
            p.particles * p.steps * N * sizeof(S),
            p.particles * p.steps * 21};
    }
};
