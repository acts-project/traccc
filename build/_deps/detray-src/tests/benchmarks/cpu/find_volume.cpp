/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Detray test include(s).
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/types.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Google include(s).
#include <benchmark/benchmark.h>

// System include(s).
#include <iostream>

// Use the detray:: namespace implicitly.
using namespace detray;

// Benchmarks the cost of searching a volume by position
void BM_FIND_VOLUMES(benchmark::State &state) {

    // This test is broken at the moment, don't run it.
    state.SkipWithError("Benchmark disabled for the toy geometry");
    return;

    // Detector configuration
    vecmem::host_memory_resource host_mr;
    toy_det_config toy_cfg{};
    toy_cfg.n_edc_layers(7u);
    auto [d, names] = build_toy_detector(host_mr, toy_cfg);

    static const unsigned int itest = 10000u;

    auto &volume_grid = d.volume_search_grid();

    const auto &axis_r = volume_grid.get_axis<axis::label::e_r>();
    const auto &axis_z = volume_grid.get_axis<axis::label::e_z>();

    // Get a rough step size from irregular axes
    auto range0 = axis_r.span();
    auto range1 = axis_z.span();

    scalar step0{(range0[1] - range0[0]) / itest};
    scalar step1{(range1[1] - range1[0]) / itest};

    std::size_t successful{0u};
    std::size_t unsuccessful{0u};

    for (auto _ : state) {
        for (unsigned int i1 = 0u; i1 < itest; ++i1) {
            for (unsigned int i0 = 0u; i0 < itest; ++i0) {
                test::vector3 rz{static_cast<scalar>(i0) * step0, 0.f,
                                 static_cast<scalar>(i1) * step1};
                const auto &v = d.volume(rz);

                benchmark::DoNotOptimize(successful);
                benchmark::DoNotOptimize(unsuccessful);
                if (v.index() == dindex_invalid) {
                    ++unsuccessful;
                } else {
                    ++successful;
                }
                benchmark::ClobberMemory();
            }
        }
    }

#ifdef DETRAY_BENCHMARK_PRINTOUTS
    std::cout << "Successful   : " << successful << std::endl;
    std::cout << "Unsuccessful : " << unsuccessful << std::endl;
#endif  // DETRAY_BENCHMARK_PRINTOUTS
}

BENCHMARK(BM_FIND_VOLUMES)
#ifdef DETRAY_BENCHMARK_MULTITHREAD
    ->ThreadRange(1, benchmark::CPUInfo::Get().num_cpus)
#endif
    ->Unit(benchmark::kMillisecond);
