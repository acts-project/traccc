/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <covfie/benchmark/propagate.hpp>
#include <covfie/benchmark/types.hpp>
#include <covfie/core/definitions.hpp>
#include <covfie/core/field_view.hpp>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

template <typename propagator_t, typename backend_t>
HOST_DEVICE inline __attribute__((always_inline)) void lorentz_step(
    const covfie::field_view<backend_t> & f,
    covfie::benchmark::lorentz_agent<3> & o,
    float s
)
{
    float lf[3];

    if constexpr (std::is_same_v<propagator_t, Euler>) {
        typename std::decay_t<decltype(f)>::output_t b =
            f.at(o.pos[0], o.pos[1], o.pos[2]);
        lf[0] = o.mom[1] * b[2] - o.mom[2] * b[1];
        lf[1] = o.mom[2] * b[0] - o.mom[0] * b[2];
        lf[2] = o.mom[0] * b[1] - o.mom[1] * b[0];
    } else {
        typename std::decay_t<decltype(f)>::output_t k1 =
            f.at(o.pos[0], o.pos[1], o.pos[2]);
        typename std::decay_t<decltype(f)>::output_t k2 = f.at(
            o.pos[0] + 0.5f * s * k1[0],
            o.pos[1] + 0.5f * s * k1[1],
            o.pos[2] + 0.5f * s * k1[2]
        );
        typename std::decay_t<decltype(f)>::output_t k3 = f.at(
            o.pos[0] + 0.5f * s * k2[0],
            o.pos[1] + 0.5f * s * k2[1],
            o.pos[2] + 0.5f * s * k2[2]
        );
        typename std::decay_t<decltype(f)>::output_t k4 = f.at(
            o.pos[0] + s * k3[0], o.pos[1] + s * k3[1], o.pos[2] + s * k3[2]
        );

        lf[0] = 0.166666667f * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
        lf[1] = 0.166666667f * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
        lf[2] = 0.166666667f * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
    }

    o.pos[0] += o.mom[0] * s;
    o.pos[1] += o.mom[1] * s;
    o.pos[2] += o.mom[2] * s;

    if (UNLIKELY(
            o.pos[0] < -9999.f || o.pos[0] > 9999.f || o.pos[1] < -9999.f ||
            o.pos[1] > 9999.f || o.pos[2] < -14999.f || o.pos[2] > 14999.f
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

    o.mom[0] += lf[0] * s;
    o.mom[1] += lf[1] * s;
    o.mom[2] += lf[2] * s;
}
