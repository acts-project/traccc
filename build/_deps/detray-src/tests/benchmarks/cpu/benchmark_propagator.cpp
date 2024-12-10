/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2020-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/containers.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/units.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/detectors/toy_metadata.hpp"
#include "detray/geometry/shapes/rectangle2D.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/tracks/tracks.hpp"
#include "detray/utils/grid/grid.hpp"

// Detray test include(s).
#include "detray/test/utils/detectors/build_toy_detector.hpp"
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"
#include "detray/test/utils/types.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

// System include(s)
#include <cstdlib>
#include <vector>

// Use the detray:: namespace implicitly.
using namespace detray;

using algebra_t = ALGEBRA_PLUGIN<detray::scalar>;

using detector_host_type = detector<toy_metadata, host_container_types>;
using detector_device_type = detector<toy_metadata, device_container_types>;

using intersection_t =
    intersection2D<typename detector_device_type::surface_type, algebra_t>;

using navigator_host_type = navigator<detector_host_type>;
using navigator_device_type = navigator<detector_device_type>;
using field_type = bfield::const_field_t;
using rk_stepper_type = rk_stepper<field_type::view_t, algebra_t>;
using actor_chain_t = actor_chain<tuple, parameter_transporter<algebra_t>,
                                  pointwise_material_interactor<algebra_t>,
                                  parameter_resetter<algebra_t>>;
using propagator_host_type =
    propagator<rk_stepper_type, navigator_host_type, actor_chain_t>;
using propagator_device_type =
    propagator<rk_stepper_type, navigator_device_type, actor_chain_t>;

enum class propagate_option {
    e_unsync = 0,
    e_sync = 1,
};

// VecMem memory resource(s)
vecmem::host_memory_resource host_mr;

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
static void BM_PROPAGATOR_CPU(benchmark::State &state) {

    std::size_t n_tracks{static_cast<std::size_t>(state.range(0)) *
                         static_cast<std::size_t>(state.range(0))};

    // Create the toy geometry and bfield
    auto [det, names] = build_toy_detector(host_mr, toy_cfg);
    test::vector3 B{0.f, 0.f, 2.f * unit<scalar>::T};
    auto bfield = bfield::create_const_field(B);

    // Create propagator
    propagation::config cfg{};
    cfg.navigation.search_window = {3u, 3u};
    propagator_host_type p{cfg};

    std::size_t total_tracks = 0;

    for (auto _ : state) {

        // TODO: use fixture to build tracks
        state.PauseTiming();

        // Get tracks
        vecmem::vector<free_track_parameters<algebra_t>> tracks(&host_mr);
        fill_tracks(tracks, n_tracks);

        total_tracks += tracks.size();

        state.ResumeTiming();

#pragma omp parallel for
        for (auto &track : tracks) {

            parameter_transporter<algebra_t>::state transporter_state{};
            pointwise_material_interactor<algebra_t>::state interactor_state{};
            parameter_resetter<algebra_t>::state resetter_state{};

            auto actor_states =
                tie(transporter_state, interactor_state, resetter_state);

            // Create the propagator state
            propagator_host_type::state p_state(track, bfield, det);

            // Run propagation
            if constexpr (opt == propagate_option::e_unsync) {
                p.propagate(p_state, actor_states);
            } else if constexpr (opt == propagate_option::e_sync) {
                p.propagate_sync(p_state, actor_states);
            }
        }
    }

    state.counters["TracksPropagated"] = benchmark::Counter(
        static_cast<double>(total_tracks), benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(BM_PROPAGATOR_CPU, propagate_option::e_unsync)
    ->Name("CPU unsync propagation")
    ->RangeMultiplier(2)
    ->Range(8, 256);
BENCHMARK_TEMPLATE(BM_PROPAGATOR_CPU, propagate_option::e_sync)
    ->Name("CPU sync propagation")
    ->RangeMultiplier(2)
    ->Range(8, 256);

BENCHMARK_MAIN();
