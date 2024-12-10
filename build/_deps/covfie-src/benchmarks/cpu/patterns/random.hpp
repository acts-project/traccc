/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <random>

#include <covfie/benchmark/pattern.hpp>
#include <covfie/core/field_view.hpp>

struct RandomFloat : covfie::benchmark::AccessPattern<RandomFloat> {
    struct parameters {
        std::size_t agents;
    };

    static constexpr std::array<std::string_view, 2> parameter_names = {"N"};

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

        std::uniform_real_distribution<float> xy_dist(-10000.f, 10000.f);
        std::uniform_real_distribution<float> z_dist(-15000.f, 15000.f);

        std::vector<covfie::benchmark::random_agent<float, 3>> objs(p.agents);

        for (std::size_t i = 0; i < objs.size(); ++i) {
            objs[i].pos[0] = xy_dist(e);
            objs[i].pos[1] = xy_dist(e);
            objs[i].pos[2] = z_dist(e);
        }

        state.ResumeTiming();

        std::size_t n = objs.size();
        for (std::size_t i = 0; i < n; ++i) {
            ::benchmark::DoNotOptimize(
                f.at(objs[i].pos[0], objs[i].pos[1], objs[i].pos[2])
            );
        }
    }

    static std::vector<std::vector<int64_t>> get_parameter_ranges()
    {
        return {{32768, 65536, 131072, 262144}};
    }

    static parameters get_parameters(benchmark::State & state)
    {
        return {static_cast<std::size_t>(state.range(0))};
    }

    template <typename S, std::size_t N>
    static covfie::benchmark::counters get_counters(const parameters & p)
    {
        return {p.agents, p.agents * N * sizeof(S), 0};
    }
};

struct RandomIntegral : covfie::benchmark::AccessPattern<RandomIntegral> {
    struct parameters {
        std::size_t agents;
    };

    static constexpr std::array<std::string_view, 2> parameter_names = {"N"};

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

        using scalar_t = typename backend_t::contravariant_input_t::scalar_t;

        std::uniform_int_distribution<scalar_t> xy_dist(0, 200);
        std::uniform_int_distribution<scalar_t> z_dist(0, 300);

        std::vector<covfie::benchmark::random_agent<scalar_t, 3>> objs(p.agents
        );

        for (std::size_t i = 0; i < objs.size(); ++i) {
            objs[i].pos[0] = xy_dist(e);
            objs[i].pos[1] = xy_dist(e);
            objs[i].pos[2] = z_dist(e);
        }

        state.ResumeTiming();

        std::size_t n = objs.size();
        for (std::size_t i = 0; i < n; ++i) {
            ::benchmark::DoNotOptimize(
                f.at(objs[i].pos[0], objs[i].pos[1], objs[i].pos[2])
            );
        }
    }

    static std::vector<std::vector<int64_t>> get_parameter_ranges()
    {
        return {{32768, 65536, 131072, 262144}};
    }

    static parameters get_parameters(benchmark::State & state)
    {
        return {static_cast<std::size_t>(state.range(0))};
    }

    template <typename S, std::size_t N>
    static covfie::benchmark::counters get_counters(const parameters & p)
    {
        return {p.agents, p.agents * N * sizeof(S), 0};
    }
};
