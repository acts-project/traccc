/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/clusterization.hpp"

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Clusterization Options";

clusterization::clusterization(po::options_description& desc)
    : m_desc{description} {

    m_desc.add_options()("target-cells-per-partition",
                         po::value(&target_cells_per_partition)
                             ->default_value(target_cells_per_partition),
                         "The number of cells to merge in a partition");
    desc.add(m_desc);
}

void clusterization::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const clusterization& opt) {

    out << ">>> " << description << " <<<\n"
        << "  Target cells per partition: " << opt.target_cells_per_partition;
    return out;
}

}  // namespace traccc::opts
