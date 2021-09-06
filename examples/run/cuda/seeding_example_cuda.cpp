/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <chrono>
#include <iomanip>
#include <iostream>

// vecmem
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

// algorithms
#include "clusterization/clusterization_algorithm.hpp"
#include "cuda/track_finding/seeding_algorithm.hpp"
#include "track_finding/seeding_algorithm.hpp"

// io
#include "io/csv.hpp"
#include "io/reader.hpp"
#include "io/writer.hpp"

int seq_run(const std::string& detector_file, const std::string& hits_dir,
            unsigned int events, bool skip_cpu) {

    // Output stats
    uint64_t n_modules = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_internal_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_cuda = 0;

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;

    // Elapsed time
    float wall_time(0);
    float hit_reading_cpu(0);
    float seeding_cpu(0);
    float seeding_cuda(0);

    traccc::seeding_algorithm sa(&host_mr);
    traccc::cuda::seeding_algorithm sa_cuda(&mng_mr);

    /*-----------------
      hit reading
      -----------------*/

    /*time*/ auto start_wall_time = std::chrono::system_clock::now();

    std::vector<traccc::host_spacepoint_container> all_spacepoints;

    // Loop over events
    for (unsigned int event = 0; event < events; ++event) {

        /*time*/ auto start_hit_reading_cpu = std::chrono::system_clock::now();

        // Read the cells from the relevant event file
        std::string event_string = "000000000";
        std::string event_number = std::to_string(event);
        event_string.replace(event_string.size() - event_number.size(),
                             event_number.size(), event_number);

        std::string io_hits_file = hits_dir + std::string("/event") +
                                   event_string + std::string("-hits.csv");

        traccc::fatras_hit_reader hreader(
            io_hits_file,
            {"particle_id", "geometry_id", "tx", "ty", "tz", "tt", "tpx", "tpy",
             "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"});
        traccc::host_spacepoint_container spacepoints_per_event =
            traccc::read_hits(hreader, host_mr);

        for (size_t i = 0; i < spacepoints_per_event.size(); i++) {
            auto& spacepoints_per_module = spacepoints_per_event.get_items()[i];

            n_spacepoints += spacepoints_per_module.size();
            n_modules++;
        }

        all_spacepoints.push_back(spacepoints_per_event);

        /*time*/ auto end_hit_reading_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_hit_reading_cpu =
            end_hit_reading_cpu - start_hit_reading_cpu;
        /*time*/ hit_reading_cpu += time_hit_reading_cpu.count();
    }

    /*-------------------
      Seeding
      --------------------*/

    for (unsigned int event = 0; event < events; ++event) {

        auto& spacepoints_per_event = all_spacepoints[event];

        // cuda

        /*time*/ auto start_seeding_cuda = std::chrono::system_clock::now();

        auto sa_cuda_result = sa_cuda(spacepoints_per_event);
        auto& seeds_cuda = sa_cuda_result.second;
        n_seeds_cuda += seeds_cuda.get_headers()[0];

        /*time*/ auto end_seeding_cuda = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_seeding_cuda =
            end_seeding_cuda - start_seeding_cuda;
        /*time*/ seeding_cuda += time_seeding_cuda.count();

        // cpu

        /*time*/ auto start_seeding_cpu = std::chrono::system_clock::now();

        traccc::host_seed_container seeds;
        traccc::host_internal_spacepoint_container internal_sp_per_event;
        if (!skip_cpu) {
            auto sa_result = sa(spacepoints_per_event);
            internal_sp_per_event = sa_result.first;
            seeds = sa_result.second;
            n_internal_spacepoints += internal_sp_per_event.total_size();
        }
        n_seeds += seeds.total_size();

        /*time*/ auto end_seeding_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_seeding_cpu =
            end_seeding_cpu - start_seeding_cpu;
        /*time*/ seeding_cpu += time_seeding_cpu.count();

        /*----------------------------------
          compare seeds from cpu and cuda
          ----------------------------------*/

        if (!skip_cpu) {
            int n_match = 0;
            for (auto seed : seeds.get_items()[0]) {
                if (std::find(seeds_cuda.get_items()[0].begin(),
                              seeds_cuda.get_items()[0].begin() +
                                  seeds_cuda.get_headers()[0],
                              seed) != seeds_cuda.get_items()[0].begin() +
                                           seeds_cuda.get_headers()[0]) {
                    n_match++;
                }
            }
            float matching_rate = float(n_match) / seeds.get_headers()[0];
            std::cout << "event " << std::to_string(event)
                      << " seed matching rate: " << matching_rate << std::endl;
        }

        /*------------
          Writer
          ------------*/

        std::string event_string = "000000000";
        std::string event_number = std::to_string(event);
        event_string.replace(event_string.size() - event_number.size(),
                             event_number.size(), event_number);

        if (!skip_cpu) {
            traccc::write_spacepoints(event, spacepoints_per_event);
            traccc::write_internal_spacepoints(event, internal_sp_per_event);
            traccc::write_seeds(event, seeds);

            /*
            traccc::multiplet_statistics_writer multiplet_stat_writer{
                "event" + event_string + "-multiplet_statistics.csv"};

            auto stats = sf.get_multiplet_stats();
            for (size_t i = 0; i < stats.size(); ++i) {
                auto stat = stats[i];
                multiplet_stat_writer.append(
                    {stat.n_spM, stat.n_mid_bot_doublets,
                     stat.n_mid_top_doublets, stat.n_triplets});
            }

            traccc::seed_statistics_writer seed_stat_writer{
                "event" + event_string + "-seed_statistics.csv"};

            auto seed_stats = sf.get_seed_stats();
            seed_stat_writer.append(
                {seed_stats.n_internal_sp, seed_stats.n_seeds});
            */
        }
    }

    /*time*/ auto end_wall_time = std::chrono::system_clock::now();
    /*time*/ std::chrono::duration<double> time_wall_time =
        end_wall_time - start_wall_time;

    /*time*/ wall_time += time_wall_time.count();

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created        " << n_internal_spacepoints
              << " internal spacepoints" << std::endl;
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

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <hit_directory> "
                     "<events> <skip_cpu>"
                  << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto hit_directory = std::string(argv[2]);
    auto events = std::atoi(argv[3]);
    bool skip_cpu = std::atoi(argv[4]);

    std::cout << "Running ./seeding_example " << detector_file << " "
              << hit_directory << " "
              << " " << events << " " << skip_cpu << std::endl;
    return seq_run(detector_file, hit_directory, events, skip_cpu);
}
