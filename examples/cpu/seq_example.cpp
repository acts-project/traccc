/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <iostream>
#include <vecmem/memory/host_memory_resource.hpp>

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/internal_spacepoint.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "geometry/pixel_segmentation.hpp"
#include "io/csv.hpp"
#include "io/reader.hpp"
#include "io/utils.hpp"

// clusterization
#include "clusterization/component_connection.hpp"
#include "clusterization/measurement_creation.hpp"
#include "clusterization/spacepoint_formation.hpp"

// seeding
#include "clusterization/clusterization_algorithm.hpp"
#include "seeding/seed_finding.hpp"
#include "seeding/spacepoint_grouping.hpp"
#include "track_finding/seeding_algorithm.hpp"

int seq_run(const std::string& detector_file, const std::string& cells_dir,
            unsigned int events) {

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Algorithms
    traccc::component_connection cc;
    traccc::measurement_creation mt;
    traccc::spacepoint_formation sp;

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;

    // Memory resource used by the EDM.
    vecmem::host_memory_resource resource;

    // Loop over events
    for (unsigned int event = 0; event < events; ++event) {

        // Read the cells from the relevant event file
        std::string io_cells_file =
            traccc::data_directory() + cells_dir + "/" +
            traccc::get_event_filename(event, "-cells.csv");
        traccc::cell_reader creader(
            io_cells_file, {"geometry_id", "hit_id", "cannel0", "channel1",
                            "activation", "time"});
        traccc::host_cell_container cells_per_event =
            traccc::read_cells(creader, resource, &surface_transforms);

        // clusterization algorithm
        traccc::clusterization_algorithm ca;
        auto ca_result = ca(cells_per_event);
        auto& measurements_per_event = ca_result.first;
        auto& spacepoints_per_event = ca_result.second;

        n_modules += cells_per_event.headers.size();
        n_cells += cells_per_event.total_size();
        n_measurements += measurements_per_event.total_size();
        n_spacepoints += spacepoints_per_event.total_size();

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

        traccc::spacepoint_grouping sg(config, grid_config);
        auto internal_sp_per_event = sg(spacepoints_per_event, &resource);

        // seed finding
        traccc::seed_finding sf(config);
        auto seeds = sf(internal_sp_per_event);

        /*------------
             Writer
          ------------*/

        traccc::measurement_writer mwriter{
            traccc::get_event_filename(event, "-measurements.csv")};
        for (size_t i = 0; i < measurements_per_event.items.size(); ++i) {
            auto measurements_per_module = measurements_per_event.items[i];
            auto module = measurements_per_event.headers[i];
            for (const auto& measurement : measurements_per_module) {
                const auto& local = measurement.local;
                mwriter.append({module.module, "", local[0], local[1], 0., 0.,
                                0., 0., 0., 0., 0., 0.});
            }
        }

        traccc::spacepoint_writer spwriter{
            traccc::get_event_filename(event, "-spacepoints.csv")};
        for (size_t i = 0; i < spacepoints_per_event.items.size(); ++i) {
            auto spacepoints_per_module = spacepoints_per_event.items[i];
            auto module = spacepoints_per_event.headers[i];

            for (const auto& spacepoint : spacepoints_per_module) {
                const auto& pos = spacepoint.global;
                spwriter.append({module, pos[0], pos[1], pos[2], 0., 0., 0.});
            }
        }

        traccc::internal_spacepoint_writer internal_spwriter{
            traccc::get_event_filename(event, "-internal_spacepoints.csv")};
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

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created " << n_spacepoints << " space points. "
              << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <cell_directory> <events>"
                  << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto cell_directory = std::string(argv[2]);
    auto events = std::atoi(argv[3]);

    std::cout << "Running ./seq_exammple " << detector_file << " "
              << cell_directory << " " << events << std::endl;
    return seq_run(detector_file, cell_directory, events);
}
