/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <benchmark/benchmark.h>
#include <boost/core/demangle.hpp>
#include <boost/mp11.hpp>

namespace covfie::benchmark {
template <typename Pattern, typename Backend>
void register_bm()
{
    using Benchmark =
        typename Pattern::template Bench<typename Backend::backend_t>;

    std::vector<std::vector<int64_t>> parameter_ranges =
        Pattern::get_parameter_ranges();

    ::benchmark::RegisterBenchmark(
        (boost::core::demangle(typeid(Pattern).name()) + "/" +
         std::string(boost::core::demangle(typeid(Backend).name())))
            .c_str(),
        [](::benchmark::State & state) {
            Benchmark::execute(state, Backend::get_field());
        }
    )
        ->ArgNames(
            {Pattern::parameter_names.begin(), Pattern::parameter_names.end()}
        )
        ->ArgsProduct({parameter_ranges.begin(), parameter_ranges.end()})
        ->UseRealTime()
        ->MeasureProcessCPUTime();
}

template <typename>
struct register_product_bm_helper_helper {
};

template <typename Pattern, typename Backend>
struct register_product_bm_helper_helper<std::pair<Pattern, Backend>> {
    static void reg()
    {
        register_bm<Pattern, Backend>();
    }
};

template <typename>
struct register_product_bm_helper {
};

template <typename... Pairs>
struct register_product_bm_helper<boost::mp11::mp_list<Pairs...>> {
    static void reg()
    {
        (register_product_bm_helper_helper<Pairs>::reg(), ...);
    }
};

template <typename PatternList, typename BackendList>
void register_product_bm()
{
    register_product_bm_helper<
        boost::mp11::mp_product<std::pair, PatternList, BackendList>>::reg();
}
}
