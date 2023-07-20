/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/device/container_h2d_copy_alg.hpp"
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/finding_input_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/propagation_options.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/seed_generator.hpp"

// detray include(s).
#include "detray/detectors/create_toy_geometry.hpp"
#include "detray/propagator/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;
namespace po = boost::program_options;

int seq_run(const traccc::finding_input_config& i_cfg,
            const traccc::propagation_options<scalar>& propagation_opts,
            const traccc::common_options& common_opts, bool run_cpu) {

    /// Type declarations
    using host_detector_type =
        detray::detector<detray::detector_registry::toy_detector, covfie::field,
                         detray::host_container_types>;
    using device_detector_type =
        detray::detector<detray::detector_registry::toy_detector,
                         covfie::field_view, detray::device_container_types>;

    using b_field_t = typename host_detector_type::bfield_type;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t, traccc::transform3,
                           detray::constrained_step<>>;
    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, host_navigator_type>;
    using device_navigator_type = detray::navigator<const device_detector_type>;
    using device_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, device_navigator_type>;

    // Output Stats
    uint64_t n_found_tracks = 0;
    uint64_t n_found_tracks_cuda = 0;
    uint64_t n_fitted_tracks = 0;
    uint64_t n_fitted_tracks_cuda = 0;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    // Performance writer
    traccc::finding_performance_writer find_performance_writer(
        traccc::finding_performance_writer::config{});

    traccc::fitting_performance_writer::config writer_cfg;
    writer_cfg.file_path = "performance_track_fitting.root";
    traccc::fitting_performance_writer fit_performance_writer(writer_cfg);

    /*****************************
     * Build a geometry
     *****************************/

    // B field value and its type
    // @TODO: Set B field as argument
    const traccc::vector3 B{0, 0, 2 * detray::unit<traccc::scalar>::T};

    // Create the toy geometry
    host_detector_type host_det =
        detray::create_toy_geometry<detray::host_container_types>(
            mng_mr,
            b_field_t(b_field_t::backend_t::configuration_t{B[0], B[1], B[2]}),
            4u, 7u);

    // Detector view object
    auto det_view = detray::get_data(host_det);

    /*****************************
     * Do the reconstruction
     *****************************/

    // Copy objects
    vecmem::cuda::copy copy;

    traccc::device::container_h2d_copy_alg<traccc::measurement_container_types>
        measurement_h2d{mr, copy};

    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        track_candidate_d2h{mr, copy};

    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        track_state_d2h{mr, copy};

    // Standard deviations for seed track parameters
    static constexpr std::array<traccc::scalar, traccc::e_bound_size> stddevs =
        {0.03 * detray::unit<traccc::scalar>::mm,
         0.03 * detray::unit<traccc::scalar>::mm,
         0.017,
         0.017,
         0.001 / detray::unit<traccc::scalar>::GeV,
         1 * detray::unit<traccc::scalar>::ns};

    // Seed generator
    traccc::seed_generator<host_detector_type> sg(host_det, stddevs);

    // Finding algorithm configuration
    typename traccc::cuda::finding_algorithm<
        rk_stepper_type, device_navigator_type>::config_type cfg;
    cfg.min_track_candidates_per_track = i_cfg.track_candidates_range[0];
    cfg.max_track_candidates_per_track = i_cfg.track_candidates_range[1];
    cfg.constrained_step_size = propagation_opts.step_constraint;

    // few tracks (~1 out of 1000 tracks) are missed when chi2_max = 15
    cfg.chi2_max = 30.f;

    // Finding algorithm object
    traccc::finding_algorithm<rk_stepper_type, host_navigator_type>
        host_finding(cfg);
    traccc::cuda::finding_algorithm<rk_stepper_type, device_navigator_type>
        device_finding(cfg, mr);

    // Fitting algorithm object
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg;
    fit_cfg.step_constraint = propagation_opts.step_constraint;

    traccc::fitting_algorithm<host_fitter_type> host_fitting(fit_cfg);
    traccc::cuda::fitting_algorithm<device_fitter_type> device_fitting(fit_cfg,
                                                                       mr);

    traccc::performance::timing_info elapsedTimes;

    // Iterate over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        // Truth Track Candidates
        traccc::event_map2 evt_map2(event, common_opts.input_directory,
                                    common_opts.input_directory,
                                    common_opts.input_directory);

        traccc::track_candidate_container_types::host truth_track_candidates =
            evt_map2.generate_truth_candidates(sg, host_mr);

        // Prepare truth seeds
        traccc::bound_track_parameters_collection_types::host seeds(mr.host);
        const unsigned int n_tracks = truth_track_candidates.size();
        for (unsigned int i_trk = 0; i_trk < n_tracks; i_trk++) {
            seeds.push_back(truth_track_candidates.at(i_trk).header);
        }

        traccc::bound_track_parameters_collection_types::buffer seeds_buffer{
            static_cast<unsigned int>(seeds.size()), mr.main};
        copy.setup(seeds_buffer);
        copy(vecmem::get_data(seeds), seeds_buffer,
             vecmem::copy::type::host_to_device);

        // Read measurements
        traccc::measurement_container_types::host measurements_per_event =
            traccc::io::read_measurements_container(
                event, common_opts.input_directory, traccc::data_format::csv,
                &host_mr);
        traccc::measurement_container_types::buffer measurements_buffer =
            measurement_h2d(traccc::get_data(measurements_per_event));

        // Instantiate output cuda containers/collections
        traccc::track_candidate_container_types::buffer
            track_candidates_cuda_buffer{{{}, *(mr.host)},
                                         {{}, *(mr.host), mr.host}};
        copy.setup(track_candidates_cuda_buffer.headers);
        copy.setup(track_candidates_cuda_buffer.items);

        // Navigation buffer
        auto navigation_buffer = detray::create_candidates_buffer(
            host_det,
            device_finding.get_config().max_num_branches_per_seed *
                seeds.size(),
            mr.main, mr.host);

        {
            traccc::performance::timer t("Track finding  (cuda)", elapsedTimes);

            // Run finding
            track_candidates_cuda_buffer =
                device_finding(det_view, navigation_buffer, measurements_buffer,
                               std::move(seeds_buffer));
        }

        traccc::track_candidate_container_types::host track_candidates_cuda =
            track_candidate_d2h(track_candidates_cuda_buffer);

        // Instantiate cuda containers/collections
        traccc::track_state_container_types::buffer track_states_cuda_buffer{
            {{}, *(mr.host)}, {{}, *(mr.host), mr.host}};

        {
            traccc::performance::timer t("Track fitting  (cuda)", elapsedTimes);

            // Run fitting
            track_states_cuda_buffer = device_fitting(
                det_view, navigation_buffer, track_candidates_cuda_buffer);
        }
        traccc::track_state_container_types::host track_states_cuda =
            track_state_d2h(track_states_cuda_buffer);

        // CPU containers
        traccc::finding_algorithm<
            rk_stepper_type, host_navigator_type>::output_type track_candidates;
        traccc::fitting_algorithm<host_fitter_type>::output_type track_states;

        if (run_cpu) {

            {
                traccc::performance::timer t("Track finding  (cpu)",
                                             elapsedTimes);

                // Run finding
                track_candidates =
                    host_finding(host_det, measurements_per_event, seeds);
            }

            {
                traccc::performance::timer t("Track fitting  (cpu)",
                                             elapsedTimes);

                // Run fitting
                track_states = host_fitting(host_det, track_candidates);
            }
        }

        if (run_cpu) {

            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;
            unsigned int n_matches = 0;
            for (unsigned int i = 0; i < track_candidates.size(); i++) {
                auto iso = traccc::details::is_same_object(
                    track_candidates.at(i).items);

                for (unsigned int j = 0; j < track_candidates_cuda.size();
                     j++) {
                    if (iso(track_candidates_cuda.at(j).items)) {
                        n_matches++;
                        break;
                    }
                }
            }
            std::cout << "Track candidate matching Rate: "
                      << float(n_matches) / track_candidates.size()
                      << std::endl;

            /// Statistics
            n_found_tracks += track_candidates.size();
            n_fitted_tracks += track_states.size();
            n_found_tracks_cuda += track_candidates_cuda.size();
            n_fitted_tracks_cuda += track_states_cuda.size();
        }

        if (i_cfg.check_performance) {
            find_performance_writer.write(traccc::get_data(track_candidates),
                                          evt_map2);

            for (unsigned int i = 0; i < track_states_cuda.size(); i++) {
                const auto& trk_states_per_track =
                    track_states_cuda.at(i).items;

                const auto& fit_info = track_states_cuda[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_info,
                                             host_det, evt_map2);
            }
        }
    }

    if (i_cfg.check_performance) {
        find_performance_writer.finalize();
        fit_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- created (cuda) " << n_found_tracks_cuda << " found tracks"
              << std::endl;
    std::cout << "- created (cuda) " << n_fitted_tracks_cuda << " fitted tracks"
              << std::endl;
    std::cout << "- created  (cpu) " << n_found_tracks << " found tracks"
              << std::endl;
    std::cout << "- created  (cpu) " << n_fitted_tracks << " fitted tracks"
              << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 1;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::common_options common_opts(desc);
    traccc::finding_input_config finding_input_cfg(desc);
    traccc::propagation_options<scalar> propagation_opts(desc);
    desc.add_options()("run_cpu", po::value<bool>()->default_value(false),
                       "run cpu tracking as well");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    finding_input_cfg.read(vm);
    propagation_opts.read(vm);
    auto run_cpu = vm["run_cpu"].as<bool>();

    std::cout << "Running " << argv[0] << " " << common_opts.input_directory
              << " " << common_opts.events << std::endl;

    return seq_run(finding_input_cfg, propagation_opts, common_opts, run_cpu);
}
