/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <random>

#include <covfie/benchmark/pattern.hpp>
#include <covfie/benchmark/propagate.hpp>
#include <covfie/core/field_view.hpp>

template <typename Order>
struct EulerPattern : covfie::benchmark::AccessPattern<EulerPattern<Order>> {
    struct parameters {
        std::size_t agents, steps, threads;
    };

    static constexpr std::array<std::string_view, 3> parameter_names = {
        "N", "S", "T"};

    template <typename backend_t>
    static void iteration(
        const parameters & p,
        const covfie::field_view<backend_t> & f,
        ::benchmark::State & state
    )
    {
        state.PauseTiming();

        std::random_device rd;
        std::minstd_rand e(rd());

        std::uniform_real_distribution<float> pos_dist(-1000.0, 1000.0);

        std::vector<covfie::benchmark::propagation_agent<3>> objs(p.agents);

        for (std::size_t i = 0; i < objs.size(); ++i) {
            objs[i].pos[0] = pos_dist(e);
            objs[i].pos[1] = pos_dist(e);
            objs[i].pos[2] = pos_dist(e);
        }

        float ss = 0.001f;

        state.ResumeTiming();

        if constexpr (std::is_same_v<Order, Deep>) {
#pragma omp parallel for num_threads(p.threads) schedule(static)
            for (std::size_t i = 0; i < p.agents; ++i) {
                for (std::size_t s = 0; s < p.steps; ++s) {
                    propagation_step<Euler>(f, objs[i], ss);
                }
            }
        } else {
#pragma omp parallel num_threads(p.threads)
            for (std::size_t s = 0; s < p.steps; ++s) {
#pragma omp for schedule(static)
                for (std::size_t i = 0; i < p.agents; ++i) {
                    propagation_step<Euler>(f, objs[i], ss);
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
            {1, 2, 4, 8, 16, 24, 32, 48}};
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
