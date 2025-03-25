/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/edm/track_parameters.hpp"
#include "traccc/fitting/kalman_filter/kalman_actor.hpp"
#include "traccc/geometry/detector.hpp"

// Performance include(s).
#include "traccc/utils/navigation_validation.hpp"

// Detray include(s).
#include <detray/detectors/bfield.hpp>
#include <detray/propagator/actors.hpp>
#include <detray/propagator/rk_stepper.hpp>

// Detray test include(s)
#include <detray/test/validation/navigation_validation_utils.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>

int nav_val_run(const traccc::opts::input_data& input_opts,
                const traccc::opts::detector& detector_opts,
                const traccc::opts::track_propagation& propagation_opts,
                std::unique_ptr<const traccc::Logger> ilogger) {

    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    using algebra_t = traccc::default_algebra;
    using detector_t = traccc::default_detector::host;
    using b_field_t =
        covfie::field<detray::bfield::const_bknd_t<traccc::scalar>>;
    using stepper_t =
        detray::rk_stepper<b_field_t::view_t, algebra_t,
                           detray::unconstrained_step<traccc::scalar>,
                           detray::stepper_rk_policy<traccc::scalar>,
                           detray::stepping::print_inspector>;

    // Memory resource used by the application.
    vecmem::host_memory_resource host_mr;

    // Set up the detector reader configuration.
    detray::io::detector_reader_config reader_cfg;
    cfg.add_file(traccc::io::get_absolute_path(detector_opts.detector_file));
    if (detector_opts.material_file.empty() == false) {
        cfg.add_file(
            traccc::io::get_absolute_path(detector_opts.material_file));
    }
    if (detector_opts.grid_file.empty() == false) {
        cfg.add_file(traccc::io::get_absolute_path(detector_opts.grid_file));
    }

    // Read the detector.
    auto [det, names] =
        detray::io::read_detector<detector_t>(host_mr, reader_cfg);

    // Geometry context
    const detector_t::geometry_context ctx{};

    // Create B field
    const vector3 B{0.f, 0.f, 2.f * traccc::unit<scalar>::T};
    auto field = detray::bfield::create_const_field<scalar>(B);

    detray::propagation::config prop_cfg{};
    if (detector.name(names) == "toy_detector") {
        prop_cfg.navigation.search_window = {3u, 3u};  //< toy detector grid cfg
    }

    // Initial track parameters
    std::vector<traccc::free_track_parameters<algebra_t>> tracks{};
    // The traces of truth hits forward and in reverse order
    std::vector<
        dvector<traccc::navigation_validator::candidate_type<detector_t>>>
        truth_traces_fw{};
    std::vector<
        dvector<traccc::navigation_validator::candidate_type<detector_t>>>
        truth_traces_bw{};

    tracks.reserve(n_events * n_tracks_per_evt);
    truth_traces_fw.reserve(n_events * n_tracks_per_evt);
    truth_traces_bw.reserve(n_events * n_tracks_per_evt);

    // Read the simulation data back in
    for (std::size_t i_event = 0u; i_event < n_events; i_event++) {
        traccc::event_data evt_data("detray_simulation/toy_detector", i_event,
                                    host_mr, false, &det, data_format::csv);

        if (evt_data.m_particle_map.empty())
            ;
        if (evt_data.m_ptc_to_meas_map.empty())
            ;
        if (evt_data.m_particle_map.size(), gen_cfg.n_tracks())
            ;

        for (const auto& [ptcl_id, ptcl] : evt_data.m_particle_map) {
            // Make a trace of detray-understandable intersections
            auto truth_trace_fw =
                traccc::navigation_validator::transcribe_to_trace(
                    ctx, det, ptcl, evt_data.m_ptc_to_meas_map,
                    detray::navigation::direction::e_forward);

            auto truth_trace_bw =
                traccc::navigation_validator::transcribe_to_trace(
                    ctx, det, ptcl, evt_data.m_ptc_to_meas_map,
                    detray::navigation::direction::e_backward);

            if (!truth_trace_fw.empty()) {
                // @TODO: Need volume grid in case of large vertex smearing
                tracks.emplace_back(ptcl.vertex, 0.f, ptcl.momentum,
                                    ptcl.charge);

                truth_traces_fw.push_back(std::move(truth_trace_fw));
                truth_traces_bw.push_back(std::move(truth_trace_bw));
            }
        }
    }

    using transporter = detray::parameter_transporter<algebra_t>;
    using interactor = detray::pointwise_material_interactor<algebra_t>;
    // using fit_actor = traccc::kalman_actor<algebra_t>;
    using resetter = detray::parameter_resetter<algebra_t>;

    // Run the navigation and compare
    constexpr bool collect_only_sensitives{true};
    constexpr bool fail_on_diff{false};
    constexpr bool verbose{false};

    b_field_t::view_t field_view = field;

    std::cout << "\n=========================" << std::endl
              << "|| FORWARD NAVIGATION  ||" << std::endl
              << "=========================\n";

    // Forward navigation
    auto nav_dir{detray::navigation::direction::e_forward};
    const auto [trk_stats_fw, n_surfaces_fw, n_miss_nav_fw, n_miss_truth_fw] =
        detray::navigation_validator::compare_to_navigation<
            stepper_t, transporter, interactor, resetter>(
            host_mr, det, names, ctx, field_view, prop_cfg, truth_traces_fw,
            tracks, nav_dir, collect_only_sensitives, fail_on_diff, verbose);

    std::cout << "\n=========================" << std::endl
              << "|| BACKWARD NAVIGATION ||" << std::endl
              << "=========================\n";

    // Backward navigation
    nav_dir = detray::navigation::direction::e_backward;
    const auto [trk_stats_bw, n_surfaces_bw, n_miss_nav_bw, n_miss_truth_bw] =
        detray::navigation_validator::compare_to_navigation<
            stepper_t, transporter, interactor, resetter>(
            host_mr, det, names, ctx, field_view, prop_cfg, truth_traces_bw,
            tracks, nav_dir, collect_only_sensitives, fail_on_diff, verbose);

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleNavValCPU", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::program_options program_opts{
        "Navigation validation on the Host",
        {detector_opts, input_opts, propagation_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    return nav_val_run(input_opts, detector_opts, propagation_opts,
                       logger->clone());
}
