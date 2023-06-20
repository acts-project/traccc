/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seed_merging/seed_merging.hpp"
#include "traccc/efficiency/nseed_performance_writer.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/seeding_input_options.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/timer.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// ACTS include(s).
#include <Acts/Definitions/Units.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>

namespace po = boost::program_options;

int seq_run(const traccc::seeding_input_config& i_cfg,
            const traccc::common_options& common_opts) {

    // Read the surface transforms
    auto surface_transforms = traccc::io::read_geometry(i_cfg.detector_file);

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    traccc::cuda::stream stream;

    vecmem::cuda::async_copy copy{stream.cudaStream()};

    traccc::cuda::seeding_algorithm sa_cuda{mr, copy, stream};
    traccc::cuda::seed_merging sm_cuda{mr, stream};

    // performance writer
    traccc::nseed_performance_writer nsd_performance_writer(
        "nseed_performance_",
        std::make_unique<traccc::simple_charged_eta_pt_cut>(2.7f, 1._GeV),
        std::make_unique<traccc::stepped_percentage>(0.6f));

    if (i_cfg.check_performance) {
        nsd_performance_writer.initialize();
    }

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::io::spacepoint_reader_output reader_output(mr.host);

        // Instantiate cuda containers/collections
        traccc::seed_collection_types::buffer seeds_cuda_buffer(0, *(mr.host));

        std::pair<vecmem::unique_alloc_ptr<traccc::nseed<20>[]>, uint32_t> merged_seeds;

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            {
                traccc::performance::timer t("Hit reading (cpu)",
                                             elapsedTimes);
                // Read the hits from the relevant event file
                traccc::io::read_spacepoints(
                    reader_output, event, common_opts.input_directory,
                    surface_transforms, common_opts.input_data_format);
            }

            auto& spacepoints_per_event = reader_output.spacepoints;

            // Copy the spacepoint data to the device.
            traccc::spacepoint_collection_types::buffer spacepoints_cuda_buffer(
                spacepoints_per_event.size(), mr.main);
            copy(vecmem::get_data(spacepoints_per_event),
                 spacepoints_cuda_buffer);
            {
                traccc::performance::timer t("Seeding (cuda)", elapsedTimes);
                // Reconstruct the spacepoints into seeds.
                seeds_cuda_buffer = sa_cuda(spacepoints_cuda_buffer);
                stream.synchronize();
            }

            // Seed merging
            {
                traccc::performance::timer t("Seed merging (cuda)", elapsedTimes);

                merged_seeds = sm_cuda(seeds_cuda_buffer);
            }
        }

        using nseed_t = std::decay_t<decltype(merged_seeds.first)::element_type>;

        std::vector<nseed_t> nseeds(merged_seeds.second);

        cudaMemcpy(nseeds.data(), merged_seeds.first.get(), merged_seeds.second * sizeof(nseed_t), cudaMemcpyDeviceToHost);

        if (i_cfg.check_performance) {
            traccc::event_map evt_map(event, i_cfg.detector_file,
                                      common_opts.input_directory,
                                      common_opts.input_directory, host_mr);

            nsd_performance_writer.register_event(
                event, nseeds.begin(), nseeds.end(),
                reader_output.spacepoints.begin(), evt_map);
        }
    }

    if (i_cfg.check_performance) {
        nsd_performance_writer.finalize();

        std::cout << nsd_performance_writer.generate_report_str();
    }

    std::cout << "==> Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::common_options common_opts(desc);
    traccc::seeding_input_config seeding_input_cfg(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    // Read options
    common_opts.read(vm);
    seeding_input_cfg.read(vm);

    std::cout << "Running " << argv[0] << " " << seeding_input_cfg.detector_file
              << " " << common_opts.input_directory << " " << common_opts.events
              << std::endl;

    return seq_run(seeding_input_cfg, common_opts);
}
