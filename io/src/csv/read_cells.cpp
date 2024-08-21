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
    bool operator()(const traccc::cell& lhs, const traccc::cell& rhs) const {
        if (lhs.module_link != rhs.module_link) {
            return lhs.module_link < rhs.module_link;
        } else if (lhs.channel1 != rhs.channel1) {
            return (lhs.channel1 < rhs.channel1);
        } else {
            return (lhs.channel0 < rhs.channel0);
        }
    }
};  // struct cell_order

/// Helper function which finds module from csv::cell in the geometry and
/// digitization config, and initializes the modules limits with the cell's
/// properties
traccc::cell_module get_module(const std::uint64_t geometry_id,
                               const traccc::geometry* geom,
                               const traccc::digitization_config* dconfig,
                               const std::uint64_t original_geometry_id) {

    traccc::cell_module result;
    result.surface_link = detray::geometry::barcode{geometry_id};

    // Find/set the 3D position of the detector module.
    if (geom != nullptr) {

        // Check if the module ID is known.
        if (!geom->contains(result.surface_link.value())) {
            throw std::runtime_error(
                "Could not find placement for geometry ID " +
                std::to_string(result.surface_link.value()));
        }

        // Set the value on the module description.
        result.placement = (*geom)[result.surface_link.value()];
    }

    // Find/set the digitization configuration of the detector module.
    if (dconfig != nullptr) {

        // Check if the module ID is known.
        const traccc::digitization_config::Iterator geo_it =
            dconfig->find(original_geometry_id);
        if (geo_it == dconfig->end()) {
            throw std::runtime_error(
                "Could not find digitization config for geometry ID " +
                std::to_string(original_geometry_id));
        }

        // Set the value on the module description.
        const auto& binning_data = geo_it->segmentation.binningData();
        assert(binning_data.size() > 0);
        result.pixel.min_corner_x = binning_data[0].min;
        result.pixel.pitch_x = binning_data[0].step;
        if (binning_data.size() > 1) {
            result.pixel.min_corner_y = binning_data[1].min;
            result.pixel.pitch_y = binning_data[1].step;
        }
        result.pixel.dimension = geo_it->dimensions;
    }

    return result;
}

std::map<std::uint64_t, std::vector<traccc::cell> > read_deduplicated_cells(
    std::string_view filename) {

    // Temporary storage for all the cells and modules.
    std::map<std::uint64_t, std::map<traccc::cell, float, ::cell_order> >
        cellMap;

    // Construct the cell reader object.
    auto reader = traccc::io::csv::make_cell_reader(filename);

    // Read all cells from input file.
    traccc::io::csv::cell iocell;
    unsigned int nduplicates = 0;
    while (reader.read(iocell)) {

        // Construct a cell object.
        const traccc::cell cell{iocell.channel0, iocell.channel1, iocell.value,
                                iocell.timestamp, 0};

        // Add the cell to the module. At this point the module link of the
        // cells is not set up correctly yet.
        auto ret = cellMap[iocell.geometry_id].insert({cell, iocell.value});
        if (ret.second == false) {
            cellMap[iocell.geometry_id].at(cell) += iocell.value;
            ++nduplicates;
        }
    }
    if (nduplicates > 0) {
        std::cout << "WARNING: @traccc::io::csv::read_cells: " << nduplicates
                  << " duplicate cells found in " << filename << std::endl;
    }

    // Create and fill the result container. With summed activation values.
    std::map<std::uint64_t, std::vector<traccc::cell> > result;
    for (const auto& [geometry_id, cells] : cellMap) {
        for (const auto& [cell, value] : cells) {
            traccc::cell summed_cell{cell};
            summed_cell.activation = value;
            result[geometry_id].push_back(summed_cell);
        }
    }

    // Return the container.
    return result;
}

std::map<std::uint64_t, std::vector<traccc::cell> > read_all_cells(
    std::string_view filename) {

    // The result container.
    std::map<std::uint64_t, std::vector<traccc::cell> > result;

    // Construct the cell reader object.
    auto reader = traccc::io::csv::make_cell_reader(filename);

    // Read all cells from input file.
    traccc::io::csv::cell iocell;
    while (reader.read(iocell)) {

        // Add the cell to the module. At this point the module link of the
        // cells is not set up correctly yet.
        result[iocell.geometry_id].push_back({iocell.channel0, iocell.channel1,
                                              iocell.value, iocell.timestamp,
                                              0});
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

void read_cells(
    cell_reader_output& out, std::string_view filename, const geometry* geom,
    const digitization_config* dconfig,
    const std::map<std::uint64_t, detray::geometry::barcode>* barcode_map,
    const bool deduplicate) {

    // Get the cells and modules into an intermediate format.
    auto cellsMap = (deduplicate ? read_deduplicated_cells(filename)
                                 : read_all_cells(filename));

    // Fill the output containers with the ordered cells and modules.
    for (const auto& [original_geometry_id, cells] : cellsMap) {
        // Modify the geometry ID of the module if a barcode map is
        // provided.
        std::uint64_t geometry_id = original_geometry_id;
        if (barcode_map != nullptr) {
            const auto it = barcode_map->find(geometry_id);
            if (it != barcode_map->end()) {
                geometry_id = it->second.value();
            } else {
                throw std::runtime_error(
                    "Could not find barcode for geometry ID " +
                    std::to_string(geometry_id));
            }
        }

        // Add the module and its cells to the output.
        out.modules.push_back(
            get_module(geometry_id, geom, dconfig, original_geometry_id));
        for (auto& cell : cells) {
            out.cells.push_back(cell);
            // Set the module link.
            out.cells.back().module_link = out.modules.size() - 1;
        }
    }
}

}  // namespace traccc::io::csv
