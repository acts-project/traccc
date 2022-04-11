/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/component_connection.hpp"
#include "traccc/clusterization/measurement_creation.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/pixel_data.hpp"
#include "traccc/io/csv.hpp"
#include "traccc/io/reader.hpp"
#include "traccc/io/utils.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Boost
#include <boost/program_options.hpp>

// OpenMP include(s).
#include <omp.h>

// System include(s).
#include <chrono>
#include <exception>
#include <iostream>

namespace po = boost::program_options;

int par_run(const std::string &detector_file, const std::string &cells_dir,
            unsigned int events) {

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Memory resource used by the EDM.
    vecmem::host_memory_resource resource;

    // Algorithms
    traccc::component_connection cc(resource);
    traccc::measurement_creation mt(resource);
    traccc::spacepoint_formation sp(resource);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t m_modules = 0;
    uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_space_points = 0;

#pragma omp parallel for reduction (+:n_cells, n_clusters, n_measurements, n_space_points, m_modules)
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
        m_modules += cells_per_event.size();

        // Output containers
        traccc::host_measurement_container measurements_per_event;
        traccc::host_spacepoint_container spacepoints_per_event;
        measurements_per_event.reserve(cells_per_event.size());
        spacepoints_per_event.reserve(cells_per_event.size());

#pragma omp parallel for
        for (std::size_t i = 0; i < cells_per_event.size(); ++i) {
            auto &module = cells_per_event.at(i).header;
            module.pixel = traccc::pixel_data{-8.425, -36.025, 0.05, 0.05};

            // The algorithmic code part: start
            traccc::host_cluster_container clusters =
                cc(cells_per_event.at(i).items, cells_per_event.at(i).header);
            for (auto &cl_id : clusters.get_headers()) {
                cl_id.is_default = false;
                cl_id.pixel = module.pixel;
            }

            traccc::host_measurement_collection measurements_per_module =
                mt(clusters, module);
            traccc::host_spacepoint_collection spacepoints_per_module =
                sp(module, measurements_per_module);
            // The algorithmnic code part: end

            n_cells += cells_per_event.at(i).items.size();
            n_clusters += clusters.size();
            n_measurements += measurements_per_module.size();
            n_space_points += spacepoints_per_module.size();

#pragma omp critical
            {
                measurements_per_event.push_back(
                    module, std::move(measurements_per_module));

                spacepoints_per_event.push_back(
                    module.module, std::move(spacepoints_per_module));
            }
        }

        traccc::measurement_writer mwriter{
            traccc::get_event_filename(event, "-measurements.csv")};
        for (size_t i = 0; i < measurements_per_event.size(); ++i) {
            auto measurements_per_module = measurements_per_event.at(i).items;
            auto module = measurements_per_event.at(i).header;
            for (const auto &measurement : measurements_per_module) {
                const auto &local = measurement.local;
                mwriter.append({module.module, "", local[0], local[1], 0., 0.,
                                0., 0., 0., 0., 0., 0.});
            }
        }

        traccc::spacepoint_writer spwriter{
            traccc::get_event_filename(event, "-spacepoints.csv")};
        for (size_t i = 0; i < spacepoints_per_event.size(); ++i) {
            auto spacepoints_per_module = spacepoints_per_event.at(i).items;
            auto module = spacepoints_per_event.at(i).header;

            for (const auto &spacepoint : spacepoints_per_module) {
                const auto &pos = spacepoint.global;
                spwriter.append({module, pos[0], pos[1], pos[2]});
            }
        }
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << m_modules
              << " modules" << std::endl;
    std::cout << "- created " << n_clusters << " clusters. " << std::endl;
    std::cout << "- created " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created " << n_space_points << " space points. "
              << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char *argv[]) {

    // Set up the program options.
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Give some help with the program's options");
    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("cell_directory", po::value<std::string>()->required(),
                       "specify the directory of cell files");
    desc.add_options()("events", po::value<int>()->required(),
                       "number of events");

    // Interpret the program options.
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Print a help message if the user asked for it.
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    // Handle any and all errors.
    try {
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << "Couldn't interpret command line options because of:\n\n"
                  << ex.what() << "\n\n"
                  << desc << std::endl;
        return 1;
    }

    auto detector_file = vm["detector_file"].as<std::string>();
    auto cell_directory = vm["cell_directory"].as<std::string>();
    auto events = vm["events"].as<int>();

    std::cout << "Running " << argv[0] << " " << detector_file << " "
              << cell_directory << " " << events << std::endl;

    auto start = std::chrono::system_clock::now();
    auto result = par_run(detector_file, cell_directory, events);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Execution time: " << diff.count() << " sec." << std::endl;
    return result;
}
