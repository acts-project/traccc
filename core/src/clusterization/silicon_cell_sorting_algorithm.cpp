/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/silicon_cell_sorting_algorithm.hpp"

// System include(s).
#include <algorithm>
#include <numeric>

namespace traccc::host {

silicon_cell_sorting_algorithm::silicon_cell_sorting_algorithm(
    vecmem::memory_resource& mr, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr(mr) {}

silicon_cell_sorting_algorithm::output_type
silicon_cell_sorting_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells_view) const {

    // Create a device container on top of the view.
    const edm::silicon_cell_collection::const_device cells{cells_view};

    // Create a vector of cell indices, which would be sorted.
    vecmem::vector<unsigned int> indices(cells.size(), &(m_mr.get()));
    std::iota(indices.begin(), indices.end(), 0u);

    // Sort the indices according to the cells.
    std::sort(indices.begin(), indices.end(),
              [&](unsigned int lhs, unsigned int rhs) {
                  return cells.at(lhs) < cells.at(rhs);
              });

    // Fill an output container with the sorted cells.
    edm::silicon_cell_collection::host result{m_mr.get()};
    for (unsigned int i : indices) {
        result.push_back(cells.at(i));
    }

    // Return the sorted cells.
    return result;
}

}  // namespace traccc::host
