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
#include <cub/cub.cuh>
#include <vecmem/utils/copy.hpp>

#define SORT_THREADS_PER_BLOCK 128

namespace traccc::cuda {
namespace kernels {
__global__ __launch_bounds__(SORT_THREADS_PER_BLOCK) void sort_cells(
    const unsigned int cells_per_thread, const unsigned int num_cells,
    const edm::silicon_cell_collection::const_view cells,
    edm::silicon_cell_collection::view new_cells) {
    using index_t = unsigned long;
    const unsigned int PER_THREAD_MAX = 32;
    using sort_t =
        cub::BlockRadixSort<index_t, SORT_THREADS_PER_BLOCK, PER_THREAD_MAX>;

    unsigned int partition_target_size = cells_per_thread * blockDim.x;
    __shared__ typename sort_t::TempStorage tmp;
    __shared__ unsigned int partition_start, partition_end;

    const edm::silicon_cell_collection::const_device cells_device(cells);
    edm::silicon_cell_collection::device new_cells_device(new_cells);

    if (threadIdx.x == 0) {
        unsigned int start = blockIdx.x * partition_target_size;
        unsigned int end = std::min(num_cells, start + partition_target_size);

        while (start != 0 && start < num_cells &&
               cells_device.module_index().at(start - 1) ==
                   cells_device.module_index().at(start)) {
            ++start;
        }

        while (end < num_cells && cells_device.module_index().at(end - 1) ==
                                      cells_device.module_index().at(end)) {
            ++end;
        }
        partition_start = start;
        partition_end = end;
        assert(partition_start <= partition_end);
    }

    __syncthreads();

    const unsigned int partition_size = partition_end - partition_start;

    index_t keys[PER_THREAD_MAX];

    for (unsigned int i = 0; i < PER_THREAD_MAX; ++i) {
        unsigned int eff = i * blockDim.x + threadIdx.x;
        keys[i] =
            eff < partition_size
                ? (static_cast<index_t>(
                       cells_device.at(partition_start + eff).module_index())
                       << 35 |
                   static_cast<index_t>(
                       cells_device.at(partition_start + eff).channel1())
                       << 24 |
                   static_cast<index_t>(
                       cells_device.at(partition_start + eff).channel0())
                       << 13 |
                   static_cast<index_t>(eff))
                : std::numeric_limits<index_t>::max();
    }

    sort_t(tmp).SortBlockedToStriped(keys);

    for (unsigned int i = 0; i < PER_THREAD_MAX; ++i) {
        unsigned int eff = i * blockDim.x + threadIdx.x;
        if (eff < partition_size) {
            new_cells_device.at(partition_start + eff) =
                cells_device.at(partition_start + (keys[i] & 0b1111111111111));
        }
    }
}
}  // namespace kernels

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

void clusterization_algorithm::sort_cells(
    const unsigned int num_cells,
    const edm::silicon_cell_collection::const_view& cells,
    edm::silicon_cell_collection::view& new_cells) const {

    const unsigned blockSize = SORT_THREADS_PER_BLOCK;
    const unsigned int cellsPerThread = 16;
    const unsigned int numBlocks =
        (num_cells + (blockSize * cellsPerThread) - 1) /
        (blockSize * cellsPerThread);

    kernels::
        sort_cells<<<numBlocks, blockSize, 0, details::get_stream(stream())>>>(
            cellsPerThread, num_cells, cells, new_cells);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

void clusterization_algorithm::ccl_kernel(
    const ccl_kernel_payload& payload) const {

    const unsigned int num_blocks =
        (payload.n_cells + (payload.config.target_partition_size()) - 1) /
        payload.config.target_partition_size();
    kernels::ccl_kernel<<<num_blocks, payload.config.threads_per_partition,
                          2 * payload.config.max_partition_size() *
                              sizeof(device::details::index_t),
                          details::get_stream(stream())>>>(
        payload.config, payload.cells, payload.det_descr, payload.measurements,
        payload.cell_links, payload.f_backup, payload.gf_backup,
        payload.adjc_backup, payload.adjv_backup, payload.backup_mutex,
        payload.disjoint_set, payload.cluster_sizes);
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
