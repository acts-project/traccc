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
    const cell_collection_types::host& cells,
    const cell_module_collection_types::host& modules) const {

    const component_connection_algorithm::output_type clusters =
        m_cc(vecmem::get_data(cells));
    const auto clusters_data = get_data(clusters);
    return m_mc(clusters_data, vecmem::get_data(modules));
}

}  // namespace traccc
