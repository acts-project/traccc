/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc {

/// Measurement creation out of clusters
///
/// This algorithm can create measurements for a single detector module
/// for all of the clusters that were identified in that one detector
/// module.
///
class measurement_creation
    : public algorithm<alt_measurement_collection_types::host(
          const cluster_container_types::host &,
          const cell_module_collection_types::host &)> {

    public:
    /// Measurement_creation algorithm constructor
    ///
    /// @param mr The memory resource to use in the algorithm
    ///
    measurement_creation(vecmem::memory_resource &mr);

    /// Callable operator for the connected component, based on one single
    /// module
    ///
    /// @param clusters Container of cells. Each subvector of cells corresponds
    /// to a cluster
    /// @param modules Collection of detector modules the clusters link to.
    ///
    /// C++20 piping interface
    ///
    /// @return a measurement collection - usually same size or sometime
    /// slightly smaller than the input
    output_type operator()(
        const cluster_container_types::host &clusters,
        const cell_module_collection_types::host &modules) const override;

    private:
    /// The memory resource used by the algorithm
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class measurement_creation

}  // namespace traccc
