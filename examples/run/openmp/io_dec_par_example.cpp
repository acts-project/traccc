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
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/io/demonstrator_edm.hpp"
#include "traccc/io/reader.hpp"
#include "traccc/io/writer.hpp"

// Boost
#include <boost/program_options.hpp>

// System include(s).
#include <chrono>
#include <iostream>

namespace po = boost::program_options;

traccc::demonstrator_result run(traccc::demonstrator_input input_data,
                                vecmem::host_memory_resource resource) {

    // Algorithms
    traccc::component_connection cc(resource);
    traccc::measurement_creation mt(resource);
    traccc::spacepoint_formation sp(resource);

    auto startAlgorithms = std::chrono::system_clock::now();

    // Output stats
    int64_t n_modules = 0;
    uint64_t n_cells = 0;
    uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_space_points = 0;

    traccc::demonstrator_result aggregated_results(input_data.size(),
                                                   &resource);

#pragma omp parallel for reduction(+:n_modules, n_cells, n_clusters, n_measurements, n_space_points)
    for (size_t event = 0; event < input_data.size(); ++event) {
        traccc::host_cell_container cells_per_event =
            input_data.operator[](event);

        traccc::host_measurement_container measurements_per_event;
        traccc::host_spacepoint_container spacepoints_per_event;
        measurements_per_event.reserve(cells_per_event.size());
        spacepoints_per_event.reserve(cells_per_event.size());

#pragma omp parallel for
        for (size_t i = 0; i < cells_per_event.size(); ++i) {
            auto &module = cells_per_event.get_headers()[i];
            module.pixel =
                traccc::pixel_segmentation{-8.425, -36.025, 0.05, 0.05};

            // The algorithmic code part: start
            traccc::cluster_collection clusters_per_module =
                cc(cells_per_event.get_items()[i],
                   cells_per_event.get_headers()[i]);
            clusters_per_module.position_from_cell = module.pixel;

            traccc::host_measurement_collection measurements_per_module =
                mt(clusters_per_module, module);
            traccc::host_spacepoint_collection spacepoints_per_module =
                sp(module, measurements_per_module);
            // The algorithmnic code part: end

            n_cells += cells_per_event.get_items()[i].size();
            n_clusters += clusters_per_module.items.size();
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

        aggregated_results[event] =
            traccc::result({measurements_per_event, spacepoints_per_event});
    }

    auto endAlgorithms = std::chrono::system_clock::now();
    std::chrono::duration<double> diffAlgo = endAlgorithms - startAlgorithms;
    std::cout << "Algorithms time: " << diffAlgo.count() << " sec."
              << std::endl;

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created " << n_clusters << " clusters. " << std::endl;
    std::cout << "- created " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created " << n_space_points << " space points. "
              << std::endl;

    return aggregated_results;
}

// The main routine
int main(int argc, char *argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()("detector_file", po::value<std::string>()->required(),
                       "specify detector file");
    desc.add_options()("cell_directory", po::value<std::string>()->required(),
                       "specify the directory of cell files");
    desc.add_options()("events", po::value<int>()->required(),
                       "number of events");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    auto detector_file = vm["detector_file"].as<std::string>();
    auto cell_directory = vm["cell_directory"].as<std::string>();
    auto events = vm["events"].as<int>();

    std::cout << "Running " << argv[0] << " " << detector_file << " "
              << cell_directory << " " << events << std::endl;

    auto start = std::chrono::system_clock::now();
    vecmem::host_memory_resource resource;
    set_default_resource(&resource);

    traccc::write(
        run(traccc::read(events, detector_file, cell_directory, resource),
            resource));

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total execution time: " << diff.count() << " sec."
              << std::endl;
}
