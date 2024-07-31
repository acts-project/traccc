/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"

namespace traccc::host {

clusterization_algorithm::clusterization_algorithm(vecmem::memory_resource& mr)
    : m_cc(mr), m_mc(mr), m_mr(mr) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_collection_types::const_view& cells_view,
    const detector_description::const_view& dd_view) const {

    const sparse_ccl_algorithm::output_type clusters = m_cc(cells_view);
    const auto clusters_data = get_data(clusters);
    return m_mc(clusters_data, dd_view);
}

}  // namespace traccc::host
