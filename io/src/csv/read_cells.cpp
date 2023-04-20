/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_cells.hpp"

#include "make_cell_reader.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace {

/// Comparator used for sorting cells. This sorting is one of the assumptions
/// made in the clusterization algorithm
const auto comp = [](const traccc::cell& c1, const traccc::cell& c2) {
    return c1.channel1 < c2.channel1;
};

/// Helper function which finds module from csv::cell in the geometry and
/// digitization config, and initializes the modules limits with the cell's
/// properties
traccc::cell_module get_module(traccc::io::csv::cell c,
                               const traccc::geometry* geom,
                               const traccc::digitization_config* dconfig) {

    traccc::cell_module result;

    result.module = c.geometry_id;

    // Find/set the 3D position of the detector module.
    if (geom != nullptr) {

        // Check if the module ID is known.
        if (!geom->contains(result.module)) {
            throw std::runtime_error(
                "Could not find placement for geometry ID " +
                std::to_string(result.module));
        }

        // Set the value on the module description.
        result.placement = (*geom)[result.module];
    }

    // Find/set the digitization configuration of the detector module.
    if (dconfig != nullptr) {

        // Check if the module ID is known.
        const traccc::digitization_config::Iterator geo_it =
            dconfig->find(result.module);
        if (geo_it == dconfig->end()) {
            throw std::runtime_error(
                "Could not find digitization config for geometry ID " +
                std::to_string(result.module));
        }

        // Set the value on the module description.
        const auto& binning_data = geo_it->segmentation.binningData();
        assert(binning_data.size() >= 2);
        result.pixel = {binning_data[0].min, binning_data[1].min,
                        binning_data[0].step, binning_data[1].step};
    }

    return result;
}

}  // namespace

namespace traccc::io::csv {

cell_reader_output read_cells(std::string_view filename, const geometry* geom,
                              const digitization_config* dconfig,
                              vecmem::memory_resource* mr) {

    // Construct the cell reader object.
    auto reader = make_cell_reader(filename);

    // Create cell counter vector.
    std::vector<unsigned int> cellCounts;
    cellCounts.reserve(5000);

    cell_module_collection_types::host result_modules;
    if (mr != nullptr) {
        result_modules = cell_module_collection_types::host{0, mr};
    } else {
        result_modules = cell_module_collection_types::host(0);
    }
    result_modules.reserve(5000);

    // Create a cell collection, which holds on to a flat list of all the cells
    // and the position of their respective cell counter & module.
    std::vector<std::pair<csv::cell, unsigned int>> allCells;
    allCells.reserve(50000);

    // Read all cells from input file.
    csv::cell iocell;
    while (reader.read(iocell)) {

        // Look for current module in cell counter vector.
        auto rit = std::find_if(result_modules.rbegin(), result_modules.rend(),
                                [&iocell](const cell_module& mod) {
                                    return mod.module == iocell.geometry_id;
                                });
        if (rit == result_modules.rend()) {
            // Add new cell and new cell counter if a new module is found
            const cell_module mod = get_module(iocell, geom, dconfig);
            allCells.push_back({iocell, result_modules.size()});
            result_modules.push_back(mod);
            cellCounts.push_back(1);
        } else {
            // Add a new cell and update cell counter if repeat module is found
            const unsigned int pos =
                std::distance(result_modules.begin(), rit.base()) - 1;
            allCells.push_back({iocell, pos});
            ++(cellCounts[pos]);
        }
    }

    // Transform the cellCounts vector into a prefix sum for accessing
    // positions in the result vector.
    std::partial_sum(cellCounts.begin(), cellCounts.end(), cellCounts.begin());

    // The total number cells.
    const unsigned int totalCells = allCells.size();

    // Construct the result collection.
    cell_collection_types::host result_cells;
    if (mr != nullptr) {
        result_cells = cell_collection_types::host{totalCells, mr};
    } else {
        result_cells = cell_collection_types::host(totalCells);
    }

    // Member "-1" of the prefix sum vector
    unsigned int nCellsZero = 0;
    // Fill the result object with the read csv cells
    for (unsigned int i = 0; i < totalCells; ++i) {
        const csv::cell& c = allCells[i].first;

        // The position of the cell counter this cell belongs to
        const unsigned int& counterPos = allCells[i].second;

        unsigned int& prefix_sum_previous =
            counterPos == 0 ? nCellsZero : cellCounts[counterPos - 1];
        result_cells[prefix_sum_previous++] = traccc::cell{
            c.channel0, c.channel1, c.value, c.timestamp, counterPos};
    }

    if (cellCounts.size() == 0) {
        return {result_cells, result_modules};
    }
    /* This is might look a bit overcomplicated, and could be made simpler by
     * having a copy of the prefix sum vector before incrementing its value when
     * filling the vector. however this seems more efficient, but requires
     * manually setting the 1st & 2nd modules instead of just the 1st.
     */

    // Sort the cells belonging to the first module.
    std::sort(result_cells.begin(), result_cells.begin() + nCellsZero, comp);
    // Sort the cells belonging to the second module.
    std::sort(result_cells.begin() + nCellsZero,
              result_cells.begin() + cellCounts[0], comp);

    // Sort cells belonging to all other modules.
    for (unsigned int i = 1; i < cellCounts.size() - 1; ++i) {
        std::sort(result_cells.begin() + cellCounts[i - 1],
                  result_cells.begin() + cellCounts[i], comp);
    }

    // Return the two collections.
    return {result_cells, result_modules};
}

}  // namespace traccc::io::csv
