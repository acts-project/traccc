/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"

namespace traccc {

clusterization_algorithm::clusterization_algorithm(vecmem::memory_resource& mr)
    : m_cc(mr), m_mc(mr), m_mr(mr) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_container_types::host& cells) const {

    return m_mc(cells, m_cc(cells));
}

}  // namespace traccc
