/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/io/csv.hpp"
#include "traccc/io/reader.hpp"
#include "traccc/io/writer.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/seeding_input_options.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>

namespace po = boost::program_options;

int seq_run(const traccc::seeding_input_config& i_cfg,
            const traccc::common_options& common_opts, bool run_cpu) {

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(i_cfg.detector_file);

    // Output stats
    uint64_t n_modules = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_cuda = 0;

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::managed_memory_resource mng_mr;

    // Elapsed time
    float wall_time(0);
    float hit_reading_cpu(0);
    float seeding_cpu(0);
    float seeding_cuda(0);
    float tp_estimating_cpu(0);
    float tp_estimating_cuda(0);

    traccc::seeding_algorithm sa(host_mr);
    traccc::track_params_estimation tp(host_mr);

    traccc::cuda::seeding_algorithm sa_cuda({mng_mr});
    traccc::cuda::track_params_estimation tp_cuda({mng_mr});

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});
    if (i_cfg.check_performance) {
        sd_performance_writer.add_cache("CPU");
        sd_performance_writer.add_cache("CUDA");
    }

    /*time*/ auto start_wall_time = std::chrono::system_clock::now();

    // Loop over events
    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {

        /*-----------------
          hit file reading
          -----------------*/

        /*time*/ auto start_hit_reading_cpu = std::chrono::system_clock::now();

        // Read the hits from the relevant event file
        traccc::spacepoint_container_types::host spacepoints_per_event =
            traccc::read_spacepoints_from_event(
                event, common_opts.input_directory,
                common_opts.input_data_format, surface_transforms, mng_mr);

        /*time*/ auto end_hit_reading_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_hit_reading_cpu =
            end_hit_reading_cpu - start_hit_reading_cpu;
        /*time*/ hit_reading_cpu += time_hit_reading_cpu.count();

        /*----------------------------
             Seeding algorithm
          ----------------------------*/

        /// CUDA

        /*time*/ auto start_seeding_cuda = std::chrono::system_clock::now();

        auto seeds_cuda_buffer =
            sa_cuda(traccc::get_data(spacepoints_per_event, &mng_mr));

        /*time*/ auto end_seeding_cuda = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_seeding_cuda =
            end_seeding_cuda - start_seeding_cuda;
        /*time*/ seeding_cuda += time_seeding_cuda.count();

        // CPU

        traccc::seeding_algorithm::output_type seeds;

        if (run_cpu) {

            /*time*/ auto start_seeding_cpu = std::chrono::system_clock::now();
            seeds = sa(spacepoints_per_event);

            /*time*/ auto end_seeding_cpu = std::chrono::system_clock::now();
            /*time*/ std::chrono::duration<double> time_seeding_cpu =
                end_seeding_cpu - start_seeding_cpu;
            /*time*/ seeding_cpu += time_seeding_cpu.count();
        }

        /*----------------------------
          Track params estimation
          ----------------------------*/

        // CUDA

        /*time*/ auto start_tp_estimating_cuda =
            std::chrono::system_clock::now();

        auto params_cuda =
            tp_cuda(traccc::get_data(spacepoints_per_event, &mng_mr),
                    seeds_cuda_buffer);

        /*time*/ auto end_tp_estimating_cuda = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_tp_estimating_cuda =
            end_tp_estimating_cuda - start_tp_estimating_cuda;
        /*time*/ tp_estimating_cuda += time_tp_estimating_cuda.count();

        // CPU

        traccc::track_params_estimation::output_type params;
        if (run_cpu) {
            /*time*/ auto start_tp_estimating_cpu =
                std::chrono::system_clock::now();

            params = tp(std::move(spacepoints_per_event), seeds);

            /*time*/ auto end_tp_estimating_cpu =
                std::chrono::system_clock::now();
            /*time*/ std::chrono::duration<double> time_tp_estimating_cpu =
                end_tp_estimating_cpu - start_tp_estimating_cpu;
            /*time*/ tp_estimating_cpu += time_tp_estimating_cpu.count();
        }

        /*----------------------------------
          compare seeds from cpu and cuda
          ----------------------------------*/

        // Copy the seeds to the host for comparisons
        vecmem::copy copy;
        traccc::host_seed_collection seeds_cuda;
        copy(seeds_cuda_buffer, seeds_cuda);

        if (run_cpu) {
            // seeding
            int n_match = 0;

            std::vector<std::array<traccc::spacepoint, 3>> sp3_vector =
                traccc::get_spacepoint_vector(seeds, spacepoints_per_event);

            std::vector<std::array<traccc::spacepoint, 3>> sp3_vector_cuda =
                traccc::get_spacepoint_vector(seeds_cuda,
                                              spacepoints_per_event);

            for (const auto& sp3 : sp3_vector) {
                if (std::find(sp3_vector_cuda.cbegin(), sp3_vector_cuda.cend(),
                              sp3) != sp3_vector_cuda.cend()) {
                    n_match++;
                }
            }

            float matching_rate = float(n_match) / seeds.size();
            std::cout << "event " << std::to_string(event) << std::endl;
            std::cout << " seed matching rate: " << matching_rate << std::endl;

            // track parameter estimation
            n_match = 0;
            for (auto& param : params) {
                if (std::find(params_cuda.begin(), params_cuda.end(), param) !=
                    params_cuda.end()) {
                    n_match++;
                }
            }
            matching_rate = float(n_match) / params.size();
            std::cout << " track parameters matching rate: " << matching_rate
                      << std::endl;
        }

        /*----------------
             Statistics
          ---------------*/

        n_spacepoints += spacepoints_per_event.total_size();
        n_seeds_cuda += seeds_cuda.size();
        n_seeds += seeds.size();

        /*------------
          Writer
          ------------*/

        if (i_cfg.check_performance) {
            traccc::event_map evt_map(event, i_cfg.detector_file,
                                      common_opts.input_directory,
                                      common_opts.input_directory, host_mr);
            sd_performance_writer.write("CUDA", seeds_cuda,
                                        spacepoints_per_event, evt_map);
            if (run_cpu) {
                sd_performance_writer.write("CPU", seeds, spacepoints_per_event,
                                            evt_map);
            }
        }
    }

    /*time*/ auto end_wall_time = std::chrono::system_clock::now();
    /*time*/ std::chrono::duration<double> time_wall_time =
        end_wall_time - start_wall_time;

    /*time*/ wall_time += time_wall_time.count();

    if (i_cfg.check_performance) {
        sd_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (cuda) " << n_seeds_cuda << " seeds" << std::endl;
    std::cout << "==> Elpased time ... " << std::endl;
    std::cout << "wall time           " << std::setw(10) << std::left
              << wall_time << std::endl;
    std::cout << "hit reading (cpu)   " << std::setw(10) << std::left
              << hit_reading_cpu << std::endl;
    std::cout << "seeding_time (cpu)  " << std::setw(10) << std::left
              << seeding_cpu << std::endl;
    std::cout << "seeding_time (cuda) " << std::setw(10) << std::left
              << seeding_cuda << std::endl;
    std::cout << "tr_par_esti_time (cpu)    " << std::setw(10) << std::left
              << tp_estimating_cpu << std::endl;
    std::cout << "tr_par_esti_time (cuda)   " << std::setw(10) << std::left
              << tp_estimating_cuda << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
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

    return seq_run(seeding_input_cfg, common_opts, run_cpu);
}
