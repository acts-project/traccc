/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::host {

/// Measurement creation out of clusters
///
/// This algorithm can create measurements for a single detector module
/// for all of the clusters that were identified in that one detector
/// module.
///
class measurement_creation_algorithm
    : public algorithm<measurement_collection_types::host(
          const cluster_container_types::const_view &,
          const silicon_detector_description::const_view &)> {

    public:
    /// Measurement_creation algorithm constructor
    ///
    /// @param mr The memory resource to use in the algorithm
    ///
    measurement_creation_algorithm(vecmem::memory_resource &mr);

    /// Callable operator for the connected component, based on one single
    /// module
    ///
    /// @param clusters_view Clusters to turn into measurements
    /// @param dd_view       The detector description
    /// @return The reconstructed measurement collection
    ///
    output_type operator()(
        const cluster_container_types::const_view &clusters_view,
        const silicon_detector_description::const_view &dd_view) const override;

    private:
    /// The memory resource used by the algorithm
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class measurement_creation_algorithm

}  // namespace traccc::host
