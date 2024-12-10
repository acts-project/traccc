/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/core/detector.hpp"
#include "detray/detectors/toy_metadata.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/volume_graph.hpp"

// Example linear algebra plugin: std::array
#include "detray/tutorial/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s)
#include <iostream>
#include <stdexcept>
#include <string>

/// Read a detector from file. For now: Read in the geometry by calling the
/// json geometry reader directly.
int main(int argc, char** argv) {

    // Input data file
    auto reader_cfg = detray::io::detector_reader_config{};
    if (argc == 2) {
        reader_cfg.add_file(argv[1]);
    } else {
        throw std::runtime_error("Please specify an input file name!");
    }

    // Read a toy detector
    using detector_t = detray::detector<>;

    // Create an empty detector to be filled
    vecmem::host_memory_resource host_mr;

    // Read the json files
    const auto [det, names] =
        detray::io::read_detector<detector_t>(host_mr, reader_cfg);

    // Print the detector volume graph
    detray::volume_graph graph(det);
    std::cout << "\nRead " << det.volumes().size() << " volumes from file "
              << reader_cfg.files()[0u] << ":\n\n"
              << graph.to_string() << std::endl;
}
