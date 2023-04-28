/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/kokkos/seeding/spacepoint_binning.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/seeding_input_options.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>

// Kokkos include(s).
#include <Kokkos_Core.hpp>

namespace po = boost::program_options;

/// Helper function that would produce a default seed-finder configuration
traccc::seedfinder_config default_seedfinder_config() {

    traccc::seedfinder_config config;
    traccc::seedfinder_config config_copy = config.toInternalUnits();
    config.highland = 13.6 * std::sqrt(config_copy.radLengthPerSeed) *
                      (1 + 0.038 * std::log(config_copy.radLengthPerSeed));
    float maxScatteringAngle = config.highland / config_copy.minPt;
    config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
    // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV
    // and millimeter
    config.pTPerHelixRadius = 300. * config_copy.bFieldInZ;
    config.minHelixDiameter2 =
        std::pow(config_copy.minPt * 2 / config.pTPerHelixRadius, 2);
    config.pT2perRadius =
        std::pow(config.highland / config.pTPerHelixRadius, 2);
    return config;
}

/// Helper function that would produce a default spacepoint grid configuration
traccc::spacepoint_grid_config default_spacepoint_grid_config() {

    traccc::seedfinder_config config = default_seedfinder_config();
    traccc::spacepoint_grid_config grid_config;
    grid_config.bFieldInZ = config.bFieldInZ;
    grid_config.minPt = config.minPt;
    grid_config.rMax = config.rMax;
    grid_config.zMax = config.zMax;
    grid_config.zMin = config.zMin;
    grid_config.deltaRMax = config.deltaRMax;
    grid_config.cotThetaMax = config.cotThetaMax;
    return grid_config;
}

int seq_run(const traccc::seeding_input_config& i_cfg,
            const traccc::common_options& common_opts, bool run_cpu) {

    // Read the surface transforms
    auto surface_transforms = traccc::io::read_geometry(i_cfg.detector_file);

    // Output stats
    uint64_t n_modules = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_kokkos = 0;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{host_mr, &host_mr};

    traccc::seeding_algorithm sa(host_mr);
    traccc::track_params_estimation tp(host_mr);

    // KOKKOS Spacepoint Binning
    traccc::kokkos::spacepoint_binning m_spacepoint_binning(
        default_seedfinder_config(), default_spacepoint_grid_config(), mr);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});
    if (i_cfg.check_performance) {
        sd_performance_writer.add_cache("CPU");
        sd_performance_writer.add_cache("KOKKOS");
    }

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        traccc::io::spacepoint_reader_output reader_output;
        traccc::seeding_algorithm::output_type seeds;
        traccc::track_params_estimation::output_type params;

        {  // Start measuring wall time
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            /*-----------------
            hit file reading
            -----------------*/
            {
                traccc::performance::timer t("Hit reading  (cpu)",
                                             elapsedTimes);
                // Read the hits from the relevant event file
                reader_output = traccc::io::read_spacepoints(
                    event, common_opts.input_directory, surface_transforms,
                    common_opts.input_data_format, &host_mr);
            }  // stop measuring hit reading timer

            traccc::spacepoint_collection_types::host& spacepoints_per_event =
                reader_output.spacepoints;

            {  // Spacepoin binning for kokkos
                traccc::performance::timer t("Spacepoint binning (kokkos)",
                                             elapsedTimes);
                m_spacepoint_binning(vecmem::get_data(spacepoints_per_event));
            }

            /*----------------------------
                Seeding algorithm
            ----------------------------*/

            // CPU

            if (run_cpu) {
                traccc::performance::timer t("Seeding  (cpu)", elapsedTimes);
                seeds = sa(spacepoints_per_event);
            }  // stop measuring seeding cpu timer

            /*----------------------------
            Track params estimation
            ----------------------------*/

            // CPU

            if (run_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(std::move(spacepoints_per_event), seeds);
            }  // stop measuring track params cpu timer

        }  // Stop measuring wall time
    }

    if (i_cfg.check_performance) {
        sd_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (kokkos) " << n_seeds_kokkos << " seeds"
              << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    // Initialise both Kokkos and GoogleTest.
    Kokkos::initialize(argc, argv);

    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::common_options common_opts(desc);
    traccc::seeding_input_config seeding_input_cfg(desc);
    desc.add_options()("run_cpu", po::value<bool>()->default_value(false),
                       "run cpu tracking as well");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    seeding_input_cfg.read(vm);
    auto run_cpu = vm["run_cpu"].as<bool>();

    std::cout << "Running " << argv[0] << " " << seeding_input_cfg.detector_file
              << " " << common_opts.input_directory << " " << common_opts.events
              << std::endl;

    int ret = seq_run(seeding_input_cfg, common_opts, run_cpu);

    // Finalise Kokkos.
    Kokkos::finalize();

    return ret;
}
