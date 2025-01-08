/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../../sanity/contiguous_on.cuh"
#include "../../sanity/ordered_on.cuh"
#include "../../utils/barrier.hpp"
#include "../../utils/cuda_error_handling.hpp"
#include "../../utils/thread_id.hpp"
#include "../../utils/utils.hpp"
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/relations.hpp"

// Project include(s)
#include "traccc/clusterization/device/ccl_kernel.hpp"

// Vecmem include(s).
#include <cstring>
#include <vecmem/utils/copy.hpp>

namespace traccc::cuda::kernels {

/// CUDA kernel for running @c traccc::device::ccl_kernel
__global__ void ccl_kernel(
    const clustering_config cfg,
    const edm::silicon_cell_collection::const_view cells_view,
    const silicon_detector_description::const_view det_descr_view,
    measurement_collection_types::view measurements_view,
    vecmem::data::vector_view<unsigned int> cell_links,
    vecmem::data::vector_view<device::details::index_t> f_backup_view,
    vecmem::data::vector_view<device::details::index_t> gf_backup_view,
    vecmem::data::vector_view<unsigned char> adjc_backup_view,
    vecmem::data::vector_view<device::details::index_t> adjv_backup_view,
    unsigned int* backup_mutex_ptr) {

    __shared__ std::size_t partition_start, partition_end;
    __shared__ std::size_t outi;
    extern __shared__ device::details::index_t shared_v[];
    vecmem::device_atomic_ref<unsigned int> backup_mutex(*backup_mutex_ptr);

    using vector_size_t =
        vecmem::data::vector_view<device::details::index_t>::size_type;

    vecmem::data::vector_view<device::details::index_t> f_view{
        static_cast<vector_size_t>(cfg.max_partition_size()), shared_v};
    vecmem::data::vector_view<device::details::index_t> gf_view{
        static_cast<vector_size_t>(cfg.max_partition_size()),
        shared_v + cfg.max_partition_size()};
    traccc::cuda::barrier barry_r;
    const details::thread_id1 thread_id;

    device::ccl_kernel(cfg, thread_id, cells_view, det_descr_view,
                       partition_start, partition_end, outi, f_view, gf_view,
                       f_backup_view, gf_backup_view, adjc_backup_view,
                       adjv_backup_view, backup_mutex, barry_r,
                       measurements_view, cell_links);
}
}  // namespace traccc::cuda::kernels
