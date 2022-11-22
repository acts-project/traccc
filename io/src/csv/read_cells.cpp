/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_cells.hpp"

#include "make_cell_reader.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <algorithm>
#include <cassert>
#include <vector>

namespace {

/// Type used for counting the number of cells per detector module
struct cell_counter {
    uint64_t module = 0;
    std::size_t nCells = 0;
};

}  // namespace

namespace traccc::io::csv {

cell_container_types::host read_cells(std::string_view filename,
                                      const geometry* geom,
                                      const digitization_config* dconfig,
                                      vecmem::memory_resource* mr) {

    // Construct the cell reader object.
    auto reader = make_cell_reader(filename);

    // Create cell counter vector.
    std::vector<cell_counter> cell_counts;
    cell_counts.reserve(5000);

    // Create a cell collection, which holds on to a flat list of all the cells.
    std::vector<csv::cell> allCells;
    allCells.reserve(50000);

    // Read all cells from input file.
    csv::cell iocell;
    while (reader.read(iocell)) {

        // Hold on to this cell.
        allCells.push_back(iocell);

        // Increment the appropriate counter.
        auto rit = std::find_if(cell_counts.rbegin(), cell_counts.rend(),
                                [&iocell](const cell_counter& cc) {
                                    return cc.module == iocell.geometry_id;
                                });
        if (rit == cell_counts.rend()) {
            cell_counts.push_back({iocell.geometry_id, 1});
        } else {
            ++(rit->nCells);
        }
    }

    // The number of modules that have cells in them.
    const std::size_t size = cell_counts.size();

    // Construct the result container, and set up its headers.
    cell_container_types::host result;
    if (mr != nullptr) {
        result = cell_container_types::host{size, mr};
    } else {
        result = cell_container_types::host{
            cell_container_types::host::header_vector{size},
            cell_container_types::host::item_vector{size}};
    }
    for (std::size_t i = 0; i < size; ++i) {

        // Make sure that we would have just the right amount of space available
        // for the cells.
        result.get_items().at(i).reserve(cell_counts[i].nCells);

        // Construct the description of the detector module.
        cell_module& module = result.get_headers().at(i);
        module.module = cell_counts[i].module;

        // Find/set the 3D position of the detector module.
        if (geom != nullptr) {

            // Check if the module ID is known.
            if (!geom->contains(module.module)) {
                throw std::runtime_error(
                    "Could not find placement for geometry ID " +
                    std::to_string(module.module));
            }

            // Set the value on the module description.
            module.placement = (*geom)[module.module];
        }

        // Find/set the digitization configuration of the detector module.
        if (dconfig != nullptr) {

            // Check if the module ID is known.
            const digitization_config::Iterator geo_it =
                dconfig->find(module.module);
            if (geo_it == dconfig->end()) {
                throw std::runtime_error(
                    "Could not find digitization config for geometry ID " +
                    std::to_string(module.module));
            }

            // Set the value on the module description.
            const auto& binning_data = geo_it->segmentation.binningData();
            assert(binning_data.size() >= 2);
            module.pixel = {binning_data[0].min, binning_data[1].min,
                            binning_data[0].step, binning_data[1].step};
        }
    }

    // Now loop over all the cells, and put them into the appropriate modules.
    std::size_t last_module_index = 0;
    for (const csv::cell& iocell : allCells) {

        // Check if this cell belongs to the same module as the last cell did.
        if (iocell.geometry_id ==
            result.get_headers().at(last_module_index).module) {

            // If so, nothing needs to be done.
        }
        // If not, then it likely belongs to the next one.
        else if ((result.size() > (last_module_index + 1)) &&
                 (iocell.geometry_id ==
                  result.get_headers().at(last_module_index + 1).module)) {

            // If so, just increment the module index by one.
            ++last_module_index;
        }
        // If not that, then look for the appropriate module with a generic
        // search.
        else {
            auto rit = std::find_if(
                result.get_headers().rbegin(), result.get_headers().rend(),
                [&iocell](const cell_module& module) {
                    return module.module == iocell.geometry_id;
                });
            assert(rit != result.get_headers().rend());
            last_module_index =
                std::distance(result.get_headers().begin(), rit.base()) - 1;
        }

        // Add the cell to the appropriate module.
        result.get_items()
            .at(last_module_index)
            .push_back({iocell.channel0, iocell.channel1, iocell.value,
                        iocell.timestamp});

        // Update the min/max values for this module.
        result.get_headers().at(last_module_index).range0[0] =
            std::min(result.get_headers().at(last_module_index).range0[0],
                     iocell.channel0);
        result.get_headers().at(last_module_index).range0[1] =
            std::max(result.get_headers().at(last_module_index).range0[1],
                     iocell.channel0);
        result.get_headers().at(last_module_index).range1[0] =
            std::min(result.get_headers().at(last_module_index).range1[0],
                     iocell.channel1);
        result.get_headers().at(last_module_index).range1[1] =
            std::max(result.get_headers().at(last_module_index).range1[1],
                     iocell.channel1);
    }

    // Do some post-processing on the cells.
    for (std::size_t i = 0; i < result.size(); ++i) {

        // Sort the cells of this module. (Not sure why this is needed. :-/)
        std::sort(result.get_items().at(i).begin(),
                  result.get_items().at(i).end(),
                  [](const traccc::cell& c1, const traccc::cell& c2) {
                      return c1.channel1 < c2.channel1;
                  });
    }

    // Return the prepared object.
    return result;
}

}  // namespace traccc::io::csv
