/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <benchmark/benchmark.h>

#include <covfie/benchmark/types.hpp>
#include <covfie/core/field.hpp>

namespace covfie::benchmark {
template <typename T>
struct AccessPattern {
    template <typename backend_t>
    struct Bench {
        static void
        execute(::benchmark::State & state, const covfie::field<backend_t> & f)
        {
            typename T::parameters params = T::get_parameters(state);
            counters counters = T::template get_counters<
                typename backend_t::covariant_output_t::scalar_t,
                backend_t::covariant_output_t::dimensions>(params);

            for (auto _ : state) {
                T::template iteration<backend_t>(params, f, state);
            }

            state.counters["AccessCount"] = ::benchmark::Counter(
                static_cast<double>(counters.access_per_it),
                ::benchmark::Counter::kIsIterationInvariant
            );
            state.counters["AccessRate"] = ::benchmark::Counter(
                static_cast<double>(counters.access_per_it),
                ::benchmark::Counter::kIsRate |
                    ::benchmark::Counter::kIsIterationInvariant
            );
            state.counters["LoadBytes"] = ::benchmark::Counter(
                static_cast<double>(counters.bytes_per_it),
                ::benchmark::Counter::kIsIterationInvariant
            );
            state.counters["LoadBandwidth"] = ::benchmark::Counter(
                static_cast<double>(counters.bytes_per_it),
                ::benchmark::Counter::kIsIterationInvariant |
                    ::benchmark::Counter::kIsRate
            );
            state.counters["ArithmeticIntensity"] =
                static_cast<double>(counters.flops_per_it) /
                static_cast<double>(counters.bytes_per_it);
        }
    };
};
}
