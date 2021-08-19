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

int seq_run(const std::string& detector_file, const std::string& cells_dir,
            unsigned int skip_events, unsigned int events, bool skip_cpu,
            bool skip_write) {
    auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
    if (env_d_d == nullptr) {
        throw std::ios_base::failure(
            "Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
    }
    auto data_directory = std::string(env_d_d) + std::string("/");

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_internal_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_cuda = 0;

    // Elapsed time
    float wall_time(0);
    float file_reading_cpu(0);
    float clusterization_cpu(0);
    float seeding_cpu(0);
    float clusterization_cuda(0);
    float seeding_cuda(0);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;

    traccc::clusterization_algorithm ca;
    traccc::cuda::seeding_algorithm sa_cuda(&mng_mr);
    traccc::seeding_algorithm sa(&host_mr);

    /*time*/ auto start_wall_time = std::chrono::system_clock::now();

    // Loop over events
    for (unsigned int event = skip_events; event < skip_events + events;
         ++event) {

        // Read the cells from the relevant event file

        /*-----------------------------
              Read the cell data
          -----------------------------*/

        /*time*/ auto start_file_reading_cpu = std::chrono::system_clock::now();

        std::string event_string = "000000000";
        std::string event_number = std::to_string(event);
        event_string.replace(event_string.size() - event_number.size(),
                             event_number.size(), event_number);

        std::string io_cells_file = cells_dir + std::string("/event") +
                                    event_string + std::string("-cells.csv");
        traccc::cell_reader creader(
            io_cells_file, {"geometry_id", "hit_id", "cannel0", "channel1",
                            "activation", "time"});

        traccc::host_cell_container cells_per_event =
            traccc::read_cells(creader, host_mr, &surface_transforms);

        /*time*/ auto end_file_reading_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_file_reading_cpu =
            end_file_reading_cpu - start_file_reading_cpu;
        /*time*/ file_reading_cpu += time_file_reading_cpu.count();

        /*-----------------------------
              Clusterization (cpu)
          -----------------------------*/

        /*time*/ auto start_clusterization_cpu =
            std::chrono::system_clock::now();
        auto ca_result = ca(cells_per_event);
        auto& measurements_per_event = ca_result.first;
        auto& spacepoints_per_event = ca_result.second;

        /*time*/ auto end_clusterization_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_clusterization_cpu =
            end_clusterization_cpu - start_clusterization_cpu;
        /*time*/ clusterization_cpu += time_clusterization_cpu.count();

        n_modules += cells_per_event.headers.size();
        n_cells += cells_per_event.total_size();
        n_measurements += measurements_per_event.total_size();
        n_spacepoints += spacepoints_per_event.total_size();

        /*----------------------------
          Seeding algorithm
          ----------------------------*/

        // cuda

        /*time*/ auto start_seeding_cuda = std::chrono::system_clock::now();

        auto sa_cuda_result = sa_cuda(spacepoints_per_event);
        auto& seeds_cuda = sa_cuda_result.second;
        n_seeds_cuda += seeds_cuda.headers[0];

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
            for (auto seed : seeds.items[0]) {
                if (std::find(
                        seeds_cuda.items[0].begin(),
                        seeds_cuda.items[0].begin() + seeds_cuda.headers[0],
                        seed) !=
                    seeds_cuda.items[0].begin() + seeds_cuda.headers[0]) {
                    n_match++;
                }
            }
            float matching_rate = float(n_match) / seeds.headers[0];
            std::cout << "event " << std::to_string(skip_events + event)
                      << " seed matching rate: " << matching_rate << std::endl;
        }

        /*------------
             Writer
          ------------*/

        if (!skip_write && !skip_cpu) {

            traccc::measurement_writer mwriter{
                std::string("event") + event_number + "-measurements.csv"};
            for (size_t i = 0; i < measurements_per_event.items.size(); ++i) {
                auto measurements_per_module = measurements_per_event.items[i];
                auto module = measurements_per_event.headers[i];
                for (const auto& measurement : measurements_per_module) {
                    const auto& local = measurement.local;
                    mwriter.append({module.module, "", local[0], local[1], 0.,
                                    0., 0., 0., 0., 0., 0., 0.});
                }
            }

            traccc::spacepoint_writer spwriter{
                std::string("event") + event_number + "-spacepoints.csv"};
            for (size_t i = 0; i < spacepoints_per_event.items.size(); ++i) {
                auto spacepoints_per_module = spacepoints_per_event.items[i];
                auto module = spacepoints_per_event.headers[i];
                for (const auto& spacepoint : spacepoints_per_module) {
                    const auto& pos = spacepoint.global;
                    spwriter.append(
                        {module, pos[0], pos[1], pos[2], 0., 0., 0.});
                }
            }

            traccc::internal_spacepoint_writer internal_spwriter{
                std::string("event") + event_number +
                "-internal_spacepoints.csv"};
            for (size_t i = 0; i < internal_sp_per_event.items.size(); ++i) {
                auto internal_sp_per_bin = internal_sp_per_event.items[i];
                auto bin = internal_sp_per_event.headers[i].global_index;

                for (const auto& internal_sp : internal_sp_per_bin) {
                    const auto& x = internal_sp.m_x;
                    const auto& y = internal_sp.m_y;
                    const auto& z = internal_sp.m_z;
                    const auto& varR = internal_sp.m_varianceR;
                    const auto& varZ = internal_sp.m_varianceZ;
                    internal_spwriter.append({bin, x, y, z, varR, varZ});
                }
            }
        }
    }

    /*time*/ auto end_wall_time = std::chrono::system_clock::now();
    /*time*/ std::chrono::duration<double> time_wall_time =
        end_wall_time - start_wall_time;

    /*time*/ wall_time += time_wall_time.count();

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created        " << n_cells << " cells           "
              << std::endl;
    std::cout << "- created        " << n_measurements << " meaurements     "
              << std::endl;
    std::cout << "- created        " << n_spacepoints << " spacepoints     "
              << std::endl;
    std::cout << "- created        " << n_internal_spacepoints
              << " internal spacepoints" << std::endl;

    std::cout << "- created (cpu)  " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (cuda) " << n_seeds_cuda << " seeds" << std::endl;
    std::cout << "==> Elpased time ... " << std::endl;
    std::cout << "wall time           " << std::setw(10) << std::left
              << wall_time << std::endl;
    std::cout << "file reading (cpu)       " << std::setw(10) << std::left
              << file_reading_cpu << std::endl;
    std::cout << "clusterization_time (cpu)" << std::setw(10) << std::left
              << clusterization_cpu << std::endl;
    std::cout << "seeding_time (cpu)       " << std::setw(10) << std::left
              << seeding_cpu << std::endl;
    std::cout << "seeding_time (cuda)      " << std::setw(10) << std::left
              << seeding_cuda << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <hit_directory> "
                     "<skip_events> <events> <skip_cpu> <skip_write>"
                  << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto hit_directory = std::string(argv[2]);
    auto skip_events = std::atoi(argv[3]);
    auto events = std::atoi(argv[4]);
    bool skip_cpu = std::atoi(argv[5]);
    bool skip_write = std::atoi(argv[6]);

    std::cout << "Running ./seq_example " << detector_file << " "
              << hit_directory << " " << skip_events << " " << events << " "
              << skip_cpu << " " << skip_write << std::endl;
    return seq_run(detector_file, hit_directory, skip_events, events, skip_cpu,
                   skip_write);
}
