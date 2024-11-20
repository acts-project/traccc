/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "write_cells.hpp"

// System include(s).
#include <fstream>
#include <stdexcept>

namespace traccc::io::csv {

void write_cells(std::string_view filename,
                 traccc::edm::silicon_cell_collection::const_view cells_view,
                 traccc::silicon_detector_description::const_view dd_view,
                 bool use_acts_geometry_id) {

    // Make sure that a valid detector description would've been given to the
    // function.
    if (dd_view.capacity() == 0u) {
        throw std::invalid_argument("Detector description must be provided");
    }

    // Open the file for writing.
    std::ofstream ofile(filename.data());
    if (!ofile.is_open()) {
        throw std::runtime_error("Could not open file " +
                                 std::string(filename));
    }

    // Create device objects.
    const edm::silicon_cell_collection::const_device cells(cells_view);
    const silicon_detector_description::const_device dd(dd_view);

    // Write the header.
    ofile << "geometry_id,measurement_id,channel0,channel1,timestamp,value\n";

    // Write out each cell.
    for (edm::silicon_cell_collection::const_device::size_type i = 0;
         i < cells.size(); ++i) {

        // Get the cell.
        const auto cell = cells.at(i);

        // Write the cell info to the file.
        ofile << (use_acts_geometry_id
                      ? dd.acts_geometry_id().at(cell.module_index())
                      : dd.geometry_id().at(cell.module_index()).value())
              << ",0," << cell.channel0() << ',' << cell.channel1() << ','
              << cell.time() << ',' << cell.activation() << '\n';
    }
}

}  // namespace traccc::io::csv
