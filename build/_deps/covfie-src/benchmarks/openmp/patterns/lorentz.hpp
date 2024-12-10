/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <random>

#include <covfie/benchmark/lorentz.hpp>
#include <covfie/benchmark/pattern.hpp>
#include <covfie/benchmark/propagate.hpp>
#include <covfie/core/field_view.hpp>

template <typename Propagator, typename Order>
struct Lorentz : covfie::benchmark::AccessPattern<Lorentz<Propagator, Order>> {
    struct parameters {
        std::size_t particles, steps, imom, threads;
    };

    static constexpr std::array<std::string_view, 4> parameter_names = {
        "N", "S", "I", "T"};

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

        float ss = 0.001f;

        std::vector<covfie::benchmark::lorentz_agent<3>> objs(p.particles);

        for (std::size_t i = 0; i < p.particles; ++i) {
            float theta = std::acos(costheta_dist(e));
            float phi = phi_dist(e);

            objs[i].pos[0] = 0.f;
            objs[i].pos[1] = 0.f;
            objs[i].pos[2] = 0.f;

            objs[i].mom[0] =
                static_cast<float>(p.imom) * std::sin(theta) * std::cos(phi);
            objs[i].mom[1] =
                static_cast<float>(p.imom) * std::sin(theta) * std::sin(phi);
            objs[i].mom[2] = static_cast<float>(p.imom) * std::cos(theta);
        }

        state.ResumeTiming();

        if constexpr (std::is_same_v<Order, Deep>) {
#pragma omp parallel for num_threads(p.threads) schedule(static)
            for (std::size_t i = 0; i < p.particles; ++i) {
                for (std::size_t s = 0; s < p.steps; ++s) {
                    lorentz_step<Propagator>(f, objs[i], ss);
                }
            }
        } else {
#pragma omp parallel num_threads(p.threads)
            for (std::size_t s = 0; s < p.steps; ++s) {
#pragma omp for schedule(static)
                for (std::size_t i = 0; i < p.particles; ++i) {
                    lorentz_step<Propagator>(f, objs[i], ss);
                }
#pragma omp barrier
            }
        }
    }

    static std::vector<std::vector<int64_t>> get_parameter_ranges()
    {
        return {
            {4096, 65536},
            {1024, 2048, 4096, 8192, 16384, 32768, 65536},
            {128,   192,   256,    384,    512,    768,    1024,  1536,  2048,
             3072,  4096,  6144,   8192,   12288,  16384,  24576, 32768, 49152,
             65536, 98304, 131072, 196608, 262144, 393216, 524288},
            {1, 2, 4, 8, 16, 24, 32, 48}};
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
