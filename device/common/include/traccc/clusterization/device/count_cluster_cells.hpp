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

// Vecmem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function used for calculating the capacities of each cluster
///
/// It results with a vector of exact cluster sizes.
/// It is necessary for allocating the appropriate aomount of memroy for
/// component connection
///
/// @param[in] globalIndex              The index of the current thread
/// @param[in] sparse_ccl_indices_view  Jagged vector that maps cells to
/// corresponding clusters
/// @param[in] cluster_prefix_sum_view  Prefix sum vector made out of number of
/// clusters in each module
/// @param[in] cells_prefix_sum_view    Prefix sum for iterating over all the
/// cells
/// @param[out] cluster_sizes_view      Container storing the number of cells
/// for each cluster
///
TRACCC_HOST_DEVICE
inline void count_cluster_cells(
    std::size_t globalIndex,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        cells_prefix_sum_view,
    vecmem::data::vector_view<unsigned int> cluster_sizes_view);

}  // namespace traccc::device

// Include the implementation
#include "traccc/clusterization/device/impl/count_cluster_cells.ipp"