/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <iostream>
#include <random>

#include <covfie/benchmark/pattern.hpp>
#include <covfie/core/definitions.hpp>
#include <covfie/core/field_view.hpp>

class Euler
{
};

class RungeKutta4
{
};

class Wide
{
};

class Deep
{
};

template <typename Propagator, typename Order>
struct Lorentz : covfie::benchmark::AccessPattern<Lorentz<Propagator, Order>> {
    struct parameters {
        std::size_t particles, steps, imom;
    };

    static constexpr std::array<std::string_view, 3> parameter_names = {
        "N", "S", "I"};

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

        std::uniform_real_distribution<float> phi_dist(0.f, 2.f * 3.1415927f);
        std::uniform_real_distribution<float> costheta_dist(-1.f, 1.f);

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
            for (std::size_t i = 0; i < p.particles; ++i) {
                for (std::size_t s = 0; s < p.steps; ++s) {
                    covfie::benchmark::lorentz_agent<3> & o = objs[i];
                    float lf[3];

                    if constexpr (std::is_same_v<Propagator, Euler>) {
                        typename std::decay_t<decltype(f)>::output_t b =
                            f.at(o.pos[0], o.pos[1], o.pos[2]);
                        lf[0] = o.mom[1] * b[2] - o.mom[2] * b[1];
                        lf[1] = o.mom[2] * b[0] - o.mom[0] * b[2];
                        lf[2] = o.mom[0] * b[1] - o.mom[1] * b[0];
                    }

                    o.pos[0] += o.mom[0] * ss;
                    o.pos[1] += o.mom[1] * ss;
                    o.pos[2] += o.mom[2] * ss;

                    if (UNLIKELY(
                            o.pos[0] < -9999.f || o.pos[0] > 9999.f ||
                            o.pos[1] < -9999.f || o.pos[1] > 9999.f ||
                            o.pos[2] < -14999.f || o.pos[2] > 14999.f
                        ))
                    {
                        if (o.pos[0] < -9999.f) {
                            o.pos[0] += 19998.f;
                        } else if (o.pos[0] > 9999.f) {
                            o.pos[0] -= 19998.f;
                        }

                        if (o.pos[1] < -9999.f) {
                            o.pos[1] += 19998.f;
                        } else if (o.pos[1] > 9999.f) {
                            o.pos[1] -= 19998.f;
                        }

                        if (o.pos[2] < -14999.f) {
                            o.pos[2] += 29998.f;
                        } else if (o.pos[2] > 14999.f) {
                            o.pos[2] -= 29998.f;
                        }
                    }

                    o.mom[0] += lf[0] * ss;
                    o.mom[1] += lf[1] * ss;
                    o.mom[2] += lf[2] * ss;
                }
            }
        } else {
            for (std::size_t s = 0; s < p.steps; ++s) {
                for (std::size_t i = 0; i < p.particles; ++i) {
                    covfie::benchmark::lorentz_agent<3> & o = objs[i];
                    float lf[3];

                    if constexpr (std::is_same_v<Propagator, Euler>) {
                        typename std::decay_t<decltype(f)>::output_t b =
                            f.at(o.pos[0], o.pos[1], o.pos[2]);
                        lf[0] = o.mom[1] * b[2] - o.mom[2] * b[1];
                        lf[1] = o.mom[2] * b[0] - o.mom[0] * b[2];
                        lf[2] = o.mom[0] * b[1] - o.mom[1] * b[0];
                    }

                    o.pos[0] += o.mom[0] * ss;
                    o.pos[1] += o.mom[1] * ss;
                    o.pos[2] += o.mom[2] * ss;

                    if (UNLIKELY(
                            o.pos[0] < -9999.f || o.pos[0] > 9999.f ||
                            o.pos[1] < -9999.f || o.pos[1] > 9999.f ||
                            o.pos[2] < -14999.f || o.pos[2] > 14999.f
                        ))
                    {
                        if (o.pos[0] < -9999.f) {
                            o.pos[0] += 19998.f;
                        } else if (o.pos[0] > 9999.f) {
                            o.pos[0] -= 19998.f;
                        }

                        if (o.pos[1] < -9999.f) {
                            o.pos[1] += 19998.f;
                        } else if (o.pos[1] > 9999.f) {
                            o.pos[1] -= 19998.f;
                        }

                        if (o.pos[2] < -14999.f) {
                            o.pos[2] += 29998.f;
                        } else if (o.pos[2] > 14999.f) {
                            o.pos[2] -= 29998.f;
                        }
                    }

                    o.mom[0] += lf[0] * ss;
                    o.mom[1] += lf[1] * ss;
                    o.mom[2] += lf[2] * ss;
                }
            }
        }
    }

    static std::vector<std::vector<int64_t>> get_parameter_ranges()
    {
        return {
            {4096, 65536},
            {1024, 2048, 4096, 8192, 16384, 32768, 65536},
            {128,   192,   256,   384,   512,   768,   1024,
             1536,  2048,  3072,  4096,  6144,  8192,  12288,
             16384, 24576, 32768, 49152, 65536, 98304, 131072}};
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
            p.particles * p.steps,
            p.particles * p.steps * N * sizeof(S),
            p.particles * p.steps * 21};
    }
};
