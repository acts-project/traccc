/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "benchmark_propagator_cuda_kernel.hpp"

// Detray test include(s).
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// Google include(s).
#include <benchmark/benchmark.h>

using namespace detray;

// VecMem memory resource(s)
vecmem::host_memory_resource host_mr;
vecmem::cuda::managed_memory_resource mng_mr;
vecmem::cuda::device_memory_resource dev_mr;
vecmem::binary_page_memory_resource bp_mng_mr(mng_mr);

// detector configuration
auto toy_cfg =
    toy_det_config{}.n_brl_layers(4u).n_edc_layers(7u).do_check(false);

void fill_tracks(vecmem::vector<free_track_parameters<algebra_t>> &tracks,
                 const std::size_t n_tracks, bool do_sort = true) {
    using scalar_t = dscalar<algebra_t>;
    using uniform_gen_t =
        detail::random_numbers<scalar_t,
                               std::uniform_real_distribution<scalar_t>>;
    using trk_generator_t =
        random_track_generator<free_track_parameters<algebra_t>, uniform_gen_t>;

    trk_generator_t::configuration trk_gen_cfg{};
    trk_gen_cfg.seed(42u);
    trk_gen_cfg.n_tracks(n_tracks);
    trk_gen_cfg.randomize_charge(true);
    trk_gen_cfg.phi_range(-constant<scalar_t>::pi, constant<scalar_t>::pi);
    trk_gen_cfg.eta_range(-3.f, 3.f);
    trk_gen_cfg.mom_range(1.f * unit<scalar_t>::GeV,
                          100.f * unit<scalar_t>::GeV);
    trk_gen_cfg.origin({0.f, 0.f, 0.f});
    trk_gen_cfg.origin_stddev({0.f * unit<scalar_t>::mm,
                               0.f * unit<scalar_t>::mm,
                               0.f * unit<scalar_t>::mm});

    // Iterate through uniformly distributed momentum directions
    for (auto traj : trk_generator_t{trk_gen_cfg}) {
        tracks.push_back(traj);
    }

    if (do_sort) {
        // Sort by theta angle
        const auto traj_comp = [](const auto &lhs, const auto &rhs) {
            constexpr auto pi_2{constant<scalar_t>::pi_2};
            return math::fabs(pi_2 - getter::theta(lhs.dir())) <
                   math::fabs(pi_2 - getter::theta(rhs.dir()));
        };

        std::ranges::sort(tracks, traj_comp);
    }
}

template <propagate_option opt>
static void BM_PROPAGATOR_CUDA(benchmark::State &state) {

    std::size_t n_tracks{static_cast<std::size_t>(state.range(0)) *
                         static_cast<std::size_t>(state.range(0))};

    // Create the toy geometry
    auto [det, names] = build_toy_detector(host_mr, toy_cfg);
    test::vector3 B{0.f, 0.f, 2.f * unit<scalar>::T};
    auto bfield = bfield::create_const_field(B);

    // vecmem copy helper object
    vecmem::cuda::copy cuda_cpy;

    // Copy detector to device
    auto det_buff = detray::get_buffer(det, dev_mr, cuda_cpy);
    auto det_view = detray::get_data(det_buff);

    std::size_t total_tracks = 0;

    for (auto _ : state) {

        state.PauseTiming();

        // Get tracks
        vecmem::vector<free_track_parameters<algebra_t>> tracks(&bp_mng_mr);
        fill_tracks(tracks, n_tracks);

        total_tracks += tracks.size();

        state.ResumeTiming();

        // Get tracks data
        auto tracks_data = vecmem::get_data(tracks);

        // Run the propagator test for GPU device
        propagator_benchmark(det_view, bfield, tracks_data, opt);
    }

    state.counters["TracksPropagated"] = benchmark::Counter(
        static_cast<double>(total_tracks), benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(BM_PROPAGATOR_CUDA, propagate_option::e_unsync)
    ->Name("CUDA unsync propagation")
    ->RangeMultiplier(2)
    ->Range(8, 256);
BENCHMARK_TEMPLATE(BM_PROPAGATOR_CUDA, propagate_option::e_sync)
    ->Name("CUDA sync propagation")
    ->RangeMultiplier(2)
    ->Range(8, 256);

BENCHMARK_MAIN();
