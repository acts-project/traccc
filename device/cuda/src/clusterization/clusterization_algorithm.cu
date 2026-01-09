/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../sanity/contiguous_on.cuh"
#include "../sanity/ordered_on.cuh"
#include "../utils/cuda_error_handling.hpp"
#include "./kernels/ccl_kernel.cuh"
#include "./kernels/reify_cluster_data.cuh"
#include "traccc/clusterization/device/ccl_kernel_definitions.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/relations.hpp"

// Vecmem include(s).
#include <cstring>
#include <vecmem/utils/copy.hpp>

namespace traccc::cuda {

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, cuda::stream& str,
    const config_type& config, std::unique_ptr<const Logger> logger)
    : device::clusterization_algorithm(mr, copy, config, std::move(logger)),
      cuda::algorithm_base(str) {}

bool clusterization_algorithm::input_is_valid(
    const edm::silicon_cell_collection::const_view& cells) const {

    return (is_contiguous_on<edm::silicon_cell_collection::const_device>(
                cell_module_projection(), mr().main, copy(), stream(), cells) &&
            is_ordered_on<edm::silicon_cell_collection::const_device>(
                channel0_major_cell_order_relation(), mr().main, copy(),
                stream(), cells));
}

void clusterization_algorithm::ccl_kernel(
    unsigned int num_cells, const config_type& config,
    const edm::silicon_cell_collection::const_view& cells,
    const silicon_detector_description::const_view& det_descr,
    edm::measurement_collection<default_algebra>::view& measurements,
    vecmem::data::vector_view<unsigned int>& cell_links,
    vecmem::data::vector_view<device::details::index_t>& f_backup,
    vecmem::data::vector_view<device::details::index_t>& gf_backup,
    vecmem::data::vector_view<unsigned char>& adjc_backup,
    vecmem::data::vector_view<device::details::index_t>& adjv_backup,
    unsigned int* backup_mutex,
    vecmem::data::vector_view<unsigned int>& disjoint_set,
    vecmem::data::vector_view<unsigned int>& cluster_sizes) const {

    const unsigned int num_blocks =
        (num_cells + (config.target_partition_size()) - 1) /
        config.target_partition_size();
    kernels::ccl_kernel<<<num_blocks, config.threads_per_partition,
                          2 * config.max_partition_size() *
                              sizeof(device::details::index_t),
                          details::get_stream(stream())>>>(
        config, cells, det_descr, measurements, cell_links, f_backup, gf_backup,
        adjc_backup, adjv_backup, backup_mutex, disjoint_set, cluster_sizes);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void clusterization_algorithm::cluster_maker_kernel(
    unsigned int num_cells,
    const vecmem::data::vector_view<unsigned int>& disjoint_set,
    edm::silicon_cluster_collection::view& cluster_data) const {

    const unsigned int num_threads = warp_size() * 16u;
    const unsigned int num_blocks = (num_cells + num_threads - 1) / num_threads;
    kernels::reify_cluster_data<<<num_blocks, num_threads, 0,
                                  details::get_stream(stream())>>>(
        disjoint_set, cluster_data);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
