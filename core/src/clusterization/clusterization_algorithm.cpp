/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"

namespace traccc::host {

clusterization_algorithm::clusterization_algorithm(
    vecmem::memory_resource& mr, std::unique_ptr<const Logger> logger)
    : messaging(logger->clone()),
      m_cc(mr, logger->cloneWithSuffix("CclAlg")),
      m_mc(mr, logger->cloneWithSuffix("MeasurementCreationAlg")),
      m_mr(mr) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const edm::silicon_cell_collection::const_view& cells_view,
    const silicon_detector_description::const_view& dd_view) const {

    const sparse_ccl_algorithm::output_type clusters = m_cc(cells_view);
    const auto clusters_data = vecmem::get_data(clusters);
    return m_mc(cells_view, clusters_data, dd_view);
}

}  // namespace traccc::host
