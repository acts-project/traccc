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

int main()
{

    auto env_d_d = std::getenv("TRACCC_TEST_DATA_DIR");
    if (env_d_d == nullptr)
    {
        throw std::ios_base::failure("Test data directory not found. Please set TRACCC_TEST_DATA_DIR.");
    }
    auto data_directory = std::string(env_d_d);

    // Read the surface transforms
    std::string detector_file = data_directory + std::string("tml_detector/trackml-detector.csv");
    traccc::surface_reader sreader(detector_file, {"geometry_id", "cx", "cy", "cz", "rot_xu", "rot_xv", "rot_xw", "rot_zu", "rot_zv", "rot_zw"});
    auto tml_barrel_transforms = traccc::read_surfaces(sreader);

    // Read the cells
    std::string cells_file = data_directory+("tml_pixel_barrel/event000000000-cells.csv");
    traccc::cell_reader creader(cells_file, {"module_id", "hit_id", "cannel0", "channel1", "activation", "time"});
    auto tml_barrel_cell_container = traccc::read_cells(creader, tml_barrel_transforms);

    // Algorithms
    traccc::component_connection cc;
    traccc::measurement_creation mt;
    traccc::spacepoint_formation sp;

    /// Output stats
    uint64_t n_cells = 0;
    uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_space_points = 0;

    for (auto &cells : tml_barrel_cell_container)
    {
        
        auto clusters = cc(cells);
        clusters.position_from_cell = traccc::pixel_segmentation{-8.425, -36.025, 0.05, 0.05};
        auto measurements = mt(clusters);
        auto spacepoints = sp(measurements);

        n_cells += cells.items.size();
        n_clusters += clusters.items.size();
        n_measurements += measurements.items.size();
        n_space_points += spacepoints.items.size();
    }

    std::cout << "- read    " << n_cells << " cells from " << tml_barrel_cell_container.size() << " modules" << std::endl;
    std::cout << "- created " << n_clusters << " clusters. " << std::endl;
    std::cout << "- created " << n_measurements << " measurements. " << std::endl;
    std::cout << "- created " << n_space_points << " space points. " << std::endl;
    return 0;
}
