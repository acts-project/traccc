/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_cells.hpp"

#include "traccc/io/csv/make_cell_reader.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

/// Comparator used for sorting cells. This sorting is one of the assumptions
/// made in the clusterization algorithm
struct cell_order {
    bool operator()(const traccc::io::csv::cell& lhs,
                    const traccc::io::csv::cell& rhs) const {
        if (lhs.channel1 != rhs.channel1) {
            return (lhs.channel1 < rhs.channel1);
        } else {
            return (lhs.channel0 < rhs.channel0);
        }
    }
};  // struct cell_order

std::map<std::uint64_t, std::vector<traccc::io::csv::cell> >
read_deduplicated_cells(std::string_view filename) {

    // Temporary storage for all the cells and modules.
    std::map<std::uint64_t,
             std::map<traccc::io::csv::cell, float, ::cell_order> >
        cellMap;

    // Construct the cell reader object.
    auto reader = traccc::io::csv::make_cell_reader(filename);

    // Read all cells from input file.
    traccc::io::csv::cell iocell;
    unsigned int nduplicates = 0;
    while (reader.read(iocell)) {

        // Add the cell to the module. At this point the module link of the
        // cells is not set up correctly yet.
        auto ret = cellMap[iocell.geometry_id].insert({iocell, iocell.value});
        if (ret.second == false) {
            cellMap[iocell.geometry_id].at(iocell) += iocell.value;
            ++nduplicates;
        }
    }
    if (nduplicates > 0) {
        std::cout << "WARNING: @traccc::io::csv::read_cells: " << nduplicates
                  << " duplicate cells found in " << filename << std::endl;
    }

    // Create and fill the result container. With summed activation values.
    std::map<std::uint64_t, std::vector<traccc::io::csv::cell> > result;
    for (const auto& [geometry_id, cells] : cellMap) {
        for (const auto& [cell, value] : cells) {
            traccc::io::csv::cell summed_cell{cell};
            summed_cell.value = value;
            result[geometry_id].push_back(summed_cell);
        }
    }

    // Return the container.
    return result;
}

std::map<std::uint64_t, std::vector<traccc::io::csv::cell> > read_all_cells(
    std::string_view filename) {

    // The result container.
    std::map<std::uint64_t, std::vector<traccc::io::csv::cell> > result;

    // Construct the cell reader object.
    auto reader = traccc::io::csv::make_cell_reader(filename);

    // Read all cells from input file.
    traccc::io::csv::cell iocell;
    while (reader.read(iocell)) {

        // Add the cell to the module. At this point the module link of the
        // cells is not set up correctly yet.
        result[iocell.geometry_id].push_back(iocell);
    }

    // Sort the cells. Deduplication or not, they do need to be sorted.
    for (auto& [_, cells] : result) {
        std::sort(cells.begin(), cells.end(), ::cell_order());
    }

    // Return the container.
    return result;
}

}  // namespace

namespace traccc::io::csv {

void read_cells(edm::silicon_cell_collection::host& cells,
                std::string_view filename,
                const silicon_detector_description::host* dd, bool deduplicate,
                bool use_acts_geometry_id) {

    // Clear the output container.
    cells.resize(0u);

    // Get the cells and modules into an intermediate format.
    auto cellsMap = (deduplicate ? read_deduplicated_cells(filename)
                                 : read_all_cells(filename));

    // If there is a detector description object, build a map of geometry IDs
    // to indices inside the detector description.
    std::map<geometry_id, unsigned int> geomIdMap;
    if (dd) {
        if (use_acts_geometry_id) {
            for (unsigned int i = 0; i < dd->acts_geometry_id().size(); ++i) {
                geomIdMap[dd->acts_geometry_id()[i]] = i;
            }
        } else {
            for (unsigned int i = 0; i < dd->geometry_id().size(); ++i) {
                geomIdMap[dd->geometry_id()[i].value()] = i;
            }
        }
    }

    // Fill the output containers with the ordered cells and modules.
    for (const auto& [geometry_id, cellz] : cellsMap) {

        // Figure out the index of the detector description object, for this
        // group of cells.
        unsigned int ddIndex = 0;
        if (dd) {
            auto it = geomIdMap.find(geometry_id);
            if (it == geomIdMap.end()) {
                throw std::runtime_error("Could not find geometry ID (" +
                                         std::to_string(geometry_id) +
                                         ") in the detector description");
            }
            ddIndex = it->second;
        }

        // Helper lambda for setting up SoA cells.
        auto assign = [ddIndex](auto soa_cell, const cell& iocell) {
            soa_cell.channel0() = iocell.channel0;
            soa_cell.channel1() = iocell.channel1;
            soa_cell.activation() = iocell.value;
            soa_cell.time() = iocell.timestamp;
            soa_cell.module_index() = ddIndex;
        };

        // Add the cells to the output.
        for (auto& cell : cellz) {
            const std::size_t cellIndex = cells.size();
            cells.resize(cellIndex + 1);
            assign(cells.at(cellIndex), cell);
        }
    }
}

}  // namespace traccc::io::csv
