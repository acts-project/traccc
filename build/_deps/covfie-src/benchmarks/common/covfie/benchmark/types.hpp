/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <benchmark/benchmark.h>

namespace covfie::benchmark {
struct counters {
    std::size_t access_per_it;
    std::size_t bytes_per_it;
    std::size_t flops_per_it;
};

template <std::size_t N>
struct lorentz_agent {
    float pos[N];
    float mom[N];
};

template <std::size_t N>
struct propagation_agent {
    float pos[N];
};

template <typename T, std::size_t N>
struct random_agent {
    T pos[N];
};
}
