/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
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
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/geometry/detector_conditions_description.hpp"
#include "traccc/geometry/detector_design_description.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Accessor for the in-shared-memory `f`/`gf` arrays used by the primary
/// (fast) CCL code path. The two logical arrays are interleaved in a single
/// buffer so cell `n` occupies elements `2n` (parent) and `2n+1`
/// (grandparent), placing each pair in one 32-bit shared-memory bank slot
/// and avoiding the 2-way bank conflict that a non-interleaved layout of
/// 16-bit indices would produce.
struct ccl_primary_accessor {
    TRACCC_HOST_DEVICE ccl_primary_accessor(details::index_t* ptr)
        : m_ptr(ptr) {}

    TRACCC_HOST_DEVICE details::index_t& f_at(unsigned int n) {
        return m_ptr[2 * n];
    }

    TRACCC_HOST_DEVICE details::index_t& gf_at(unsigned int n) {
        return m_ptr[2 * n + 1];
    }

    private:
    details::index_t* m_ptr;
};

/// Accessor for the global-memory `f`/`gf` fallback buffers used when a
/// partition is too large to fit in the primary shared-memory buffer. The
/// elements are stored as `fallback_index_t` (32-bit) rather than `index_t`
/// (16-bit) because this is the code path whose partition size
/// may exceed the `unsigned short` range.
struct ccl_backup_accessor {
    TRACCC_HOST_DEVICE ccl_backup_accessor(details::fallback_index_t* f_ptr,
                                           details::fallback_index_t* gf_ptr)
        : m_f_ptr(f_ptr), m_gf_ptr(gf_ptr) {}

    TRACCC_HOST_DEVICE details::fallback_index_t& f_at(unsigned int n) {
        return m_f_ptr[n];
    }

    TRACCC_HOST_DEVICE details::fallback_index_t& gf_at(unsigned int n) {
        return m_gf_ptr[n];
    }

    private:
    details::fallback_index_t *m_f_ptr, *m_gf_ptr;
};

/// Function which reads raw detector cells and turns them into measurements.
///
/// @param[in] cfg clustering configuration
/// @param[in] thread_id a thread identifier object
/// @param[in] cells_view    collection of cells
/// @param[in] det_descr_view The detectorsegmentation description
/// @param[in] det_cond_view The detector conditions description
/// @param partition_start    partition start point for this thread block
/// @param partition_end      partition end point for this thread block
/// @param outi               number of measurements for this partition
/// @param fgf_ptr pointer to shared memory that will house both f and gf
/// @param f_backup_view global memory alternative to fgf_ptr for cases in
///     which that array is not large enough
/// @param gf_backup_view global memory alternative to fgf_ptr for cases in
///     which that array is not large enough
/// @param adjc_backup_view global memory alternative to the adjacent cell
///     count vector
/// @param adjv_backup_view global memory alternative to the cell adjacency
///     matrix fragment storage
/// @param backup_mutex mutex lock to mediate control over the backup global
///     memory data structures.
/// @param[out] disjoint_set_view Array of unsigned integers of
///     length $|cells|$ to which an integer is written identifying the
///     measurement index to which each cell belongs.
/// @param[out] cluster_size_view Array of unsigned integers of
///     size $|cells|$; the first $N$ elements - where $N$ is the number of
///     output measurements - will have their value set to the size of the
///     corresponding measurement.
/// @param barrier  A generic object for block-wide synchronisation
/// @param[out] measurements_view collection of measurements
/// @param[out] cell_links    collection of links to measurements each cell is
/// put into
template <device::concepts::barrier barrier_t,
          device::concepts::thread_id1 thread_id_t>
TRACCC_DEVICE inline void ccl_kernel(
    const clustering_config cfg, const thread_id_t& thread_id,
    const edm::silicon_cell_collection::const_view& cells_view,
    const detector_design_description::const_view& det_descr_view,
    const detector_conditions_description::const_view& det_cond_view,
    std::size_t& partition_start, std::size_t& partition_end, std::size_t& outi,
    details::index_t* fgf_ptr,
    vecmem::data::vector_view<details::fallback_index_t> f_backup_view,
    vecmem::data::vector_view<details::fallback_index_t> gf_backup_view,
    vecmem::data::vector_view<unsigned char> adjc_backup_view,
    vecmem::data::vector_view<details::fallback_index_t> adjv_backup_view,
    vecmem::device_atomic_ref<uint32_t> backup_mutex,
    vecmem::data::vector_view<unsigned int> disjoint_set_view,
    vecmem::data::vector_view<unsigned int> cluster_size_view,
    const barrier_t& barrier,
    edm::measurement_collection::view measurements_view,
    vecmem::data::vector_view<unsigned int> cell_links);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/ccl_kernel.ipp"
