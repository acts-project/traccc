/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/kokkos/seeding/spacepoint_binning.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
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

int seq_run(const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& /*finding_opts*/,
            const traccc::opts::track_propagation& /*propagation_opts*/,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts) {

    // Read the surface transforms
    auto [surface_transforms, _] =
        traccc::io::read_geometry(detector_opts.detector_file);

    // Output stats
    uint64_t n_modules = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_kokkos = 0;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    traccc::memory_resource mr{host_mr, &host_mr};

    traccc::seeding_algorithm sa(seeding_opts.seedfinder,
                                 {seeding_opts.seedfinder},
                                 seeding_opts.seedfilter, host_mr);
    traccc::track_params_estimation tp(host_mr);

    // KOKKOS Spacepoint Binning
    traccc::kokkos::spacepoint_binning m_spacepoint_binning(
        seeding_opts.seedfinder, {seeding_opts.seedfinder}, mr);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        traccc::io::spacepoint_reader_output reader_output(&host_mr);
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
                traccc::io::read_spacepoints(
                    reader_output, event, input_opts.directory,
                    surface_transforms, input_opts.format);
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

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Seeding  (cpu)", elapsedTimes);
                seeds = sa(spacepoints_per_event);
            }  // stop measuring seeding cpu timer

            /*----------------------------
            Track params estimation
            ----------------------------*/

            // CPU

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(std::move(spacepoints_per_event), seeds,
                            {0.f, 0.f, seeding_opts.seedfinder.bFieldInZ});
            }  // stop measuring track params cpu timer

        }  // Stop measuring wall time
    }

    if (performance_opts.run) {
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
    traccc::opts::detector detector_opts{desc};
    traccc::opts::input_data input_opts{desc};
    traccc::opts::track_seeding seeding_opts{desc};
    traccc::opts::track_finding finding_opts{desc};
    traccc::opts::track_propagation propagation_opts{desc};
    traccc::opts::performance performance_opts{desc};
    traccc::opts::accelerator accelerator_opts{desc};

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    detector_opts.read(vm);
    input_opts.read(vm);
    seeding_opts.read(vm);
    finding_opts.read(vm);
    propagation_opts.read(vm);
    performance_opts.read(vm);
    accelerator_opts.read(vm);

    // Tell the user what's happening.
    std::cout << "\nRunning the tracking chain using Kokkos\n\n"
              << detector_opts << "\n"
              << input_opts << "\n"
              << seeding_opts << "\n"
              << finding_opts << "\n"
              << propagation_opts << "\n"
              << performance_opts << "\n"
              << accelerator_opts << "\n"
              << std::endl;

    const int ret =
        seq_run(seeding_opts, finding_opts, propagation_opts, input_opts,
                detector_opts, performance_opts, accelerator_opts);

    // Finalise Kokkos.
    Kokkos::finalize();

    return ret;
}
