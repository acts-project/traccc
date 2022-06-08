/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/seeding/seed_finding.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Library include(s).
#include "traccc/cuda/seeding/seed_selecting.hpp"
#include "traccc/cuda/seeding/triplet_counting.hpp"
#include "traccc/cuda/seeding/triplet_finding.hpp"
#include "traccc/cuda/seeding/weight_updating.hpp"

// Project include(s).
#include "traccc/device/get_prefix_sum.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/seeding/device/count_doublets.hpp"
#include "traccc/seeding/device/find_doublets.hpp"
#include "traccc/seeding/device/make_doublet_buffers.hpp"
#include "traccc/seeding/device/make_doublet_counter_buffer.hpp"

// VecMem include(s).
#include "vecmem/utils/copy.hpp"

// System include(s).
#include <algorithm>
#include <vector>

namespace traccc::cuda {
namespace kernels {

/// CUDA kernel for running @c traccc::device::count_doublets
__global__ void count_doublets(
    seedfinder_config config, sp_grid_const_view sp_grid,
    vecmem::data::vector_view<const device::prefix_sum_element_t> sp_prefix_sum,
    device::doublet_counter_container_types::view doublet_counter) {

    device::count_doublets(threadIdx.x + blockIdx.x * blockDim.x, config,
                           sp_grid, sp_prefix_sum, doublet_counter);
}

/// CUDA kernel for running @c traccc::device::find_doublets
__global__ void find_doublets(
    seedfinder_config config, sp_grid_const_view sp_grid,
    device::doublet_counter_container_types::const_view doublet_counter,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        doublet_prefix_sum,
    doublet_container_view mb_doublets, doublet_container_view mt_doublets) {

    device::find_doublets(threadIdx.x + blockIdx.x * blockDim.x, config,
                          sp_grid, doublet_counter, doublet_prefix_sum,
                          mb_doublets, mt_doublets);
}

}  // namespace kernels

seed_finding::seed_finding(const seedfinder_config& config,
                           vecmem::memory_resource& mr)
    : m_seedfinder_config(config), m_mr(mr) {}

seed_finding::output_type seed_finding::operator()(
    const spacepoint_container_types::host& spacepoints,
    const sp_grid_const_view& g2_view) const {

    // Helper object for the data management.
    vecmem::copy copy;

    // Get the sizes of the grid
    auto grid_sizes = copy.get_sizes(g2_view._data_view);

    // Get the prefix sum for the spacepoint grid.
    const device::prefix_sum_t sp_grid_prefix_sum =
        device::get_prefix_sum(grid_sizes, m_mr.get());

    // Set up the doublet counter buffer.
    device::doublet_counter_container_types::buffer doublet_counter_buffer =
        device::make_doublet_counter_buffer(grid_sizes, copy, m_mr.get());

    // Calculate the number of threads and thread blocks to run the doublet
    // counting kernel for.
    const unsigned int nDoubletCountThreads = WARP_SIZE * 2;
    const unsigned int nDoubletCountBlocks =
        sp_grid_prefix_sum.size() / nDoubletCountThreads + 1;

    // Count the number of doublets that we need to produce.
    kernels::count_doublets<<<nDoubletCountBlocks, nDoubletCountThreads>>>(
        m_seedfinder_config, g2_view, vecmem::get_data(sp_grid_prefix_sum),
        doublet_counter_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Get the summary values per bin.
    vecmem::vector<device::doublet_counter_header> doublet_counts(
        &(m_mr.get()));
    copy(doublet_counter_buffer.headers, doublet_counts);

    // Set up the doublet buffers.
    device::doublet_buffer_pair doublet_buffers =
        device::make_doublet_buffers(doublet_counter_buffer, copy, m_mr.get());

    // Get the prefix sum for the doublet counter buffer.
    const device::prefix_sum_t doublet_prefix_sum =
        device::get_prefix_sum(doublet_counter_buffer.items, m_mr.get(), copy);

    // Calculate the number of threads and thread blocks to run the doublet
    // finding kernel for.
    const unsigned int nDoubletFindThreads = WARP_SIZE * 2;
    const unsigned int nDoubletFindBlocks =
        doublet_prefix_sum.size() / nDoubletFindThreads + 1;

    // Find all of the spacepoint doublets.
    kernels::find_doublets<<<nDoubletFindBlocks, nDoubletFindThreads>>>(
        m_seedfinder_config, g2_view, doublet_counter_buffer,
        vecmem::get_data(doublet_prefix_sum), doublet_buffers.middleBottom,
        doublet_buffers.middleTop);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    vecmem::vector<doublet_per_bin> mb_headers(&m_mr.get());
    copy(doublet_buffers.middleBottom.headers, mb_headers);

    // Create triplet counter container buffer
    unsigned int nbins = g2_view._data_view.m_size;
    std::vector<std::size_t> mb_buffer_sizes(doublet_counts.size());
    std::transform(
        doublet_counts.begin(), doublet_counts.end(), mb_buffer_sizes.begin(),
        [](const device::doublet_counter_header& dc) { return dc.m_nMidBot; });
    triplet_counter_container_buffer tcc_buffer{{nbins, m_mr.get()},
                                                {mb_buffer_sizes, m_mr.get()}};
    copy.setup(tcc_buffer.headers);

    // Run triplet counting
    traccc::cuda::triplet_counting(
        m_seedfinder_config, mb_headers, g2_view, doublet_counter_buffer,
        doublet_buffers.middleBottom, doublet_buffers.middleTop, tcc_buffer,
        m_mr.get());

    // Take header of the triplet counter container buffer into host
    vecmem::vector<triplet_counter_per_bin> tcc_headers(&m_mr.get());
    copy(tcc_buffer.headers, tcc_headers);

    // Fill the size vector for triplet container
    std::vector<size_t> n_triplets_per_bin;
    n_triplets_per_bin.reserve(nbins);
    for (const auto& h : tcc_headers) {
        n_triplets_per_bin.push_back(h.n_triplets);
    }

    // Create triplet container buffer
    triplet_container_buffer tc_buffer{{nbins, m_mr.get()},
                                       {n_triplets_per_bin, m_mr.get()}};
    copy.setup(tc_buffer.headers);

    // Run triplet finding
    traccc::cuda::triplet_finding(
        m_seedfinder_config, m_seedfilter_config, tcc_headers, g2_view,
        doublet_counter_buffer, doublet_buffers.middleBottom,
        doublet_buffers.middleTop, tcc_buffer, tc_buffer, m_mr.get());

    // Take header of the triplet container buffer into host
    vecmem::vector<triplet_per_bin> tc_headers(&m_mr.get());
    copy(tc_buffer.headers, tc_headers);

    // Run weight updating
    traccc::cuda::weight_updating(m_seedfilter_config, tc_headers, g2_view,
                                  tcc_buffer, tc_buffer, m_mr.get());

    // Get the number of seeds (triplets)
    auto n_triplets = std::accumulate(n_triplets_per_bin.begin(),
                                      n_triplets_per_bin.end(), 0);

    // Create seed buffer
    vecmem::data::vector_buffer<seed> seed_buffer(n_triplets, 0, m_mr.get());
    copy.setup(seed_buffer);

    // Run seed selecting
    traccc::cuda::seed_selecting(
        m_seedfilter_config, doublet_counts, spacepoints, g2_view,
        doublet_counter_buffer, tcc_buffer, tc_buffer, seed_buffer, m_mr.get());

    // Take seed buffer into seed collection
    host_seed_collection seed_collection(&m_mr.get());
    copy(seed_buffer, seed_collection);

    return seed_collection;
}

}  // namespace traccc::cuda
