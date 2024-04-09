/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/clusterization.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

clusterization::clusterization() : interface("Clusterization Options") {

    m_desc.add_options()(
        "target-cells-per-partition",
        boost::program_options::value(&target_cells_per_partition)
            ->default_value(target_cells_per_partition),
        "The number of cells to merge in a partition");
}

std::ostream& clusterization::print_impl(std::ostream& out) const {

    out << "  Target cells per partition: " << target_cells_per_partition;
    return out;
}

}  // namespace traccc::opts
