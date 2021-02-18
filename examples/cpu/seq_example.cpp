/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "geometry/pixel_segmentation.hpp"
#include "algorithms/component_connection.hpp"
#include "algorithms/measurement_creation.hpp"
#include "algorithms/spacepoint_formation.hpp"
#include "csv/csv_io.hpp"

#include <iostream>

int seq_run(const std::string& detector_file, const std::string& cells_dir, unsigned int events)
{
    auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
    if (env_d_d == nullptr)
    {
        throw std::ios_base::failure("Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
    }
    auto data_directory = std::string(env_d_d);

    // Read the surface transforms
    std::string io_detector_file = data_directory + detector_file;
    traccc::surface_reader sreader(io_detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv", "rot_xw", "rot_zu", "rot_zv", "rot_zw"});
    auto surface_transforms = traccc::read_surfaces(sreader);

    // Algorithms
    traccc::component_connection cc;
    traccc::measurement_creation mt;
    traccc::spacepoint_formation sp;

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_space_points = 0;

    // Loop over events
    for (unsigned int event = 0; event < events; ++event){

        // Read the cells from the relevant event file
        std::string event_string = "000000000";
        std::string event_number = std::to_string(event);
        event_string.replace(event_string.size()-event_number.size(), event_number.size(), event_number);

        std::string io_cells_file = data_directory+cells_dir+std::string("/event")+event_string+std::string("-cells.csv");
        traccc::cell_reader creader(io_cells_file, {"geometry_id", "hit_id", "cannel0", "channel1", "activation", "time"});
        auto cell_container = traccc::read_cells(creader, surface_transforms);

        // Output containers
        std::vector<traccc::measurement_collection> measurement_container;
        std::vector<traccc::spacepoint_collection> spacepoint_container;
        measurement_container.reserve(cell_container.size());
        spacepoint_container.reserve(cell_container.size());

        for (auto &cells : cell_container)
        {
            // The algorithmic code part: start
            auto clusters = cc(cells);
            clusters.position_from_cell = traccc::pixel_segmentation{-8.425, -36.025, 0.05, 0.05};
            auto measurements = mt(clusters);
            auto spacepoints = sp(measurements);
            // The algorithmnic code part: end

            n_cells += cells.items.size();
            n_clusters += clusters.items.size();
            n_measurements += measurements.items.size();
            n_space_points += spacepoints.items.size();

            measurement_container.push_back(std::move(measurements));
            spacepoint_container.push_back(std::move(spacepoints));
        }
    
        traccc::measurement_writer mwriter{std::string("event")+event_number+"-measurements.csv"};
        for (const auto& measurement_collection : measurement_container){
            auto module = measurement_collection.module;
            for (const auto& measurement : measurement_collection.items){
                const auto& local = measurement.local;
                mwriter.append({ module, local[0], local[1], 0., 0.});
            }
        }

        traccc::spacepoint_writer spwriter{std::string("event")+event_number+"-spacepoints.csv"};
        for (const auto& spacepoint_collection : spacepoint_container){
            auto module = spacepoint_collection.module;
            for (const auto& spacepoint : spacepoint_collection.items){
                const auto& pos = spacepoint.global;
                spwriter.append({ module, pos[0], pos[1], pos[2], 0., 0., 0.});
            }
        }

    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_cells << " modules" << std::endl;
    std::cout << "- created " << n_clusters << " clusters. " << std::endl;
    std::cout << "- created " << n_measurements << " measurements. " << std::endl;
    std::cout << "- created " << n_space_points << " space points. " << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char *argv[])
{
    if (argc < 4){
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./seq_example <detector_file> <cell_directory> <events>" << std::endl;
        return -1;
    }

    auto detector_file = std::string(argv[1]);
    auto cell_directory = std::string(argv[2]);
    auto events = std::atoi(argv[3]);

    std::cout << "Running ./seq_exammple " << detector_file << " " << cell_directory << " " << events << std::endl;
    return seq_run(detector_file, cell_directory, events);
}
