/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function used for the filling the cluster container with corresponding cells
///
/// The output is the cluster container with module indices as headers and
/// clusters of cells as items Since the headers are module_idx, and not
/// cluster_idx, there can be multiple same module_idx next to each other
///
/// @param[in] globalIndex              The index for the current thread
/// @param[in] cells_view               The cells for each module
/// @param[in] sparse_ccl_indices_view  Jagged vector that maps cells to
/// corresponding clusters
/// @param[in] cluster_prefix_sum_view  Prefix sum vector made out of number of
/// clusters in each module
/// @param[in] cells_prefix_sum_view    Prefix sum for iterating over all the
/// cells
/// @param[out] clusters_view           Container storing the cells for every
/// cluster
///
TRACCC_HOST_DEVICE
inline void connect_components(
    std::size_t globalIndex, const cell_container_types::const_view& cells_view,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    cluster_container_types::view clusters_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/connect_components.ipp"