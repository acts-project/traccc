/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// edm
#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/internal_spacepoint.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "geometry/pixel_segmentation.hpp"

// seeding
#include "seeding/seed_finding.hpp"
#include "seeding/spacepoint_grouping.hpp"

// io
#include "csv/csv_io.hpp"

// vecmem
#include <vecmem/memory/host_memory_resource.hpp>

// std
#include <chrono>
#include <iomanip>

int seq_run(const std::string& detector_file, const std::string& hits_dir,
            unsigned int events) {
    // Read the surface transforms
    std::string io_detector_file = detector_file;
    traccc::surface_reader sreader(
        io_detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv",
                           "rot_xw", "rot_zu", "rot_zv", "rot_zw"});
    auto surface_transforms = traccc::read_surfaces(sreader);

    // Output stats
    uint64_t n_modules = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_internal_spacepoints = 0;
    uint64_t n_doublets = 0;
    uint64_t n_seeds = 0;

    // Memory resource used by the EDM.
    vecmem::host_memory_resource resource;

    // Elapsed time
    float binning_cpu(0);
    float seeding_cpu(0);

    // Loop over events
    for (unsigned int event = 0; event < events; ++event) {
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
            traccc::read_hits(hreader, resource);

        for (size_t i = 0; i < spacepoints_per_event.headers.size(); i++) {
            auto& spacepoints_per_module = spacepoints_per_event.items[i];

            n_spacepoints += spacepoints_per_module.size();
            n_modules++;
        }

        /*-------------------
             Seed finding
          -------------------*/

        // Seed finder config
        traccc::seedfinder_config config;
        // silicon detector max
        config.rMax = 160.;
        config.deltaRMin = 5.;
        config.deltaRMax = 160.;
        config.collisionRegionMin = -250.;
        config.collisionRegionMax = 250.;
        // config.zMin = -2800.; // this value introduces redundant bins without
        // any spacepoints config.zMax = 2800.;
        config.zMin = -1186.;
        config.zMax = 1186.;
        config.maxSeedsPerSpM = 5;
        // 2.7 eta
        config.cotThetaMax = 7.40627;
        config.sigmaScattering = 1.00000;

        config.minPt = 500.;
        config.bFieldInZ = 0.00199724;

        config.beamPos = {-.5, -.5};
        config.impactMax = 10.;

        config.highland = 13.6 * std::sqrt(config.radLengthPerSeed) *
                          (1 + 0.038 * std::log(config.radLengthPerSeed));
        float maxScatteringAngle = config.highland / config.minPt;
        config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
        // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV
        // and millimeter
        // TODO: change using ACTS units
        config.pTPerHelixRadius = 300. * config.bFieldInZ;
        config.minHelixDiameter2 =
            std::pow(config.minPt * 2 / config.pTPerHelixRadius, 2);
        config.pT2perRadius =
            std::pow(config.highland / config.pTPerHelixRadius, 2);

        // setup spacepoint grid config
        traccc::spacepoint_grid_config grid_config;
        grid_config.bFieldInZ = config.bFieldInZ;
        grid_config.minPt = config.minPt;
        grid_config.rMax = config.rMax;
        grid_config.zMax = config.zMax;
        grid_config.zMin = config.zMin;
        grid_config.deltaRMax = config.deltaRMax;
        grid_config.cotThetaMax = config.cotThetaMax;

        // create internal spacepoints grouped in bins
        /*time*/ auto start_binning_cpu = std::chrono::system_clock::now();

        traccc::spacepoint_grouping sg(config, grid_config);
        auto internal_sp_per_event = sg(spacepoints_per_event, &resource);

        /*time*/ auto end_binning_cpu = std::chrono::system_clock::now();
        /*time*/ std::chrono::duration<double> time_binning_cpu =
            end_binning_cpu - start_binning_cpu;

        /*time*/ binning_cpu += time_binning_cpu.count();

        /*-----------------------
          seed finding -- cpu
          -----------------------*/
        /*time*/ auto start_seeding_cpu = std::chrono::system_clock::now();

        traccc::seed_finding sf(config);
        auto seeds = sf(internal_sp_per_event);

        /*time*/ auto end_seeding_cpu = std::chrono::system_clock::now();
        //n_seeds += seeds.size();
	n_seeds += seeds.items[0].size();

        /*time*/ std::chrono::duration<double> time_seeding_cpu =
            end_seeding_cpu - start_seeding_cpu;
        /*time*/ seeding_cpu += time_seeding_cpu.count();

        for (size_t i = 0; i < internal_sp_per_event.headers.size(); ++i) {
            n_internal_spacepoints += internal_sp_per_event.items[i].size();
        }

        /*------------
          Writer
          ------------*/

        traccc::spacepoint_writer spwriter{std::string("event") + event_number +
                                           "-spacepoints.csv"};
        for (size_t i = 0; i < spacepoints_per_event.items.size(); ++i) {
            auto spacepoints_per_module = spacepoints_per_event.items[i];
            auto module = spacepoints_per_event.headers[i];

            for (const auto& spacepoint : spacepoints_per_module) {
                const auto& pos = spacepoint.global;
                spwriter.append({module, pos[0], pos[1], pos[2], 0., 0., 0.});
            }
        }

        traccc::internal_spacepoint_writer internal_spwriter{
            std::string("event") + event_number + "-internal_spacepoints.csv"};
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

        traccc::seed_writer sd_writer{std::string("event") + event_number +
                                      "-seeds.csv"};
        //for (size_t i = 0; i < seeds.size(); ++i) {
	for (auto seed: seeds.items[0]) {
                auto weight = seed.weight;
                auto z_vertex = seed.z_vertex;
                auto spB = seed.spB;
                auto spM = seed.spM;
                auto spT = seed.spT;

            sd_writer.append({weight, z_vertex, spB.x(), spB.y(), spB.z(), 0, 0,
                              spM.x(), spM.y(), spM.z(), 0, 0, spT.x(), spT.y(),
                              spT.z(), 0, 0});
        }

        traccc::multiplet_statistics_writer multiplet_stat_writer{
            std::string("event") + event_number + "-multiplet_statistics.csv"};

        auto stats = sf.get_multiplet_stats();
        for (size_t i = 0; i < stats.size(); ++i) {
            auto stat = stats[i];
            multiplet_stat_writer.append({stat.n_spM, stat.n_mid_bot_doublets,
                                          stat.n_mid_top_doublets,
                                          stat.n_triplets});
        }

        traccc::seed_statistics_writer seed_stat_writer{
            std::string("event") + event_number + "-seed_statistics.csv"};

        auto seed_stats = sf.get_seed_stats();
        seed_stat_writer.append({seed_stats.n_internal_sp, seed_stats.n_seeds});
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_spacepoints << " spacepoints from "
              << n_modules << " modules" << std::endl;
    std::cout << "- created " << n_internal_spacepoints
              << " internal spacepoints" << std::endl;
    std::cout << "- created " << n_seeds << " seeds" << std::endl;
    std::cout << "==> Elpased time ... " << std::endl;
    std::cout << "binning_time       " << std::setw(10) << std::left
              << binning_cpu << std::endl;
    std::cout << "seeding_time       " << std::setw(10) << std::left
              << seeding_cpu << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <hit_directory> <events>"
                  << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto hit_directory = std::string(argv[2]);
    auto events = std::atoi(argv[3]);

    std::cout << "Running ./seeding_example " << detector_file << " "
              << hit_directory << " " << events << std::endl;
    return seq_run(detector_file, hit_directory, events);
}
