/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/definitions/hints.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function which reads raw detector cells and turns them into measurements.
///
/// @param[in] cfg clustering configuration
/// @param[in] thread_id a thread identifier object
/// @param[in] cells_view    collection of cells
/// @param[in] det_descr_view The detector description
/// @param partition_start    partition start point for this thread block
/// @param partition_end      partition end point for this thread block
/// @param outi               number of measurements for this partition
/// @param f_view  array of "parent" indices for all cells in this partition
/// @param gf_view array of "grandparent" indices for all cells in this
///                partition
/// @param f_backup_view global memory alternative to `f_view` for cases in
///     which that array is not large enough
/// @param gf_backup_view global memory alternative to `gf_view` for cases in
///     which that array is not large enough
/// @param adjc_backup_view global memory alternative to the adjacent cell
///     count vector
/// @param adjv_backup_view global memory alternative to the cell adjacency
///     matrix fragment storage
/// @param backup_mutex mutex lock to mediate control over the backup global
///     memory data structures.
/// @param barrier  A generic object for block-wide synchronisation
/// @param[out] measurements_view collection of measurements
/// @param[out] cell_links    collection of links to measurements each cell is
/// put into
template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t>
TRACCC_DEVICE inline void ccl_kernel(
    const clustering_config cfg, const thread_id_t& thread_id,
    const edm::silicon_cell_collection::const_view& cells_view,
    const silicon_detector_description::const_view& det_descr_view,
    std::size_t& partition_start, std::size_t& partition_end, std::size_t& outi,
    vecmem::data::vector_view<details::index_t> f_view,
    vecmem::data::vector_view<details::index_t> gf_view,
    vecmem::data::vector_view<details::index_t> f_backup_view,
    vecmem::data::vector_view<details::index_t> gf_backup_view,
    vecmem::data::vector_view<unsigned char> adjc_backup_view,
    vecmem::data::vector_view<details::index_t> adjv_backup_view,
    vecmem::device_atomic_ref<uint32_t> backup_mutex, const barrier_t& barrier,
    measurement_collection_types::view measurements_view,
    vecmem::data::vector_view<unsigned int> cell_links);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/ccl_kernel.ipp"
