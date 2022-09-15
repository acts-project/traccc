/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/seeding/seed_finding.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s).
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/device/get_prefix_sum.hpp"
#include "traccc/device/make_prefix_sum_buffer.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/seeding/device/count_doublets.hpp"
#include "traccc/seeding/device/count_triplets.hpp"
#include "traccc/seeding/device/find_doublets.hpp"
#include "traccc/seeding/device/find_triplets.hpp"
#include "traccc/seeding/device/make_doublet_buffers.hpp"
#include "traccc/seeding/device/make_doublet_counter_buffer.hpp"
#include "traccc/seeding/device/make_triplet_buffer.hpp"
#include "traccc/seeding/device/make_triplet_counter_buffer.hpp"
#include "traccc/seeding/device/select_seeds.hpp"
#include "traccc/seeding/device/update_triplet_weights.hpp"

// VecMem include(s).
#include "vecmem/utils/cuda/copy.hpp"

// System include(s).
#include <algorithm>
#include <vector>

namespace traccc::cuda {
namespace kernels {

/// CUDA kernel for running @c traccc::device::fill_prefix_sum
__global__ void fill_prefix_sum(
    vecmem::data::vector_view<const device::prefix_sum_size_t> sizes_view,
    vecmem::data::vector_view<device::prefix_sum_element_t> ps_view) {

    device::fill_prefix_sum(threadIdx.x + blockIdx.x * blockDim.x, sizes_view,
                            ps_view);
}

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

/// CUDA kernel for running @c traccc::device::count_triplets
__global__ void count_triplets(
    seedfinder_config config, sp_grid_const_view sp_grid,
    device::doublet_counter_container_types::const_view doublet_counter_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        doublet_prefix_sum,
    doublet_container_view mb_doublets, doublet_container_view mt_doublets,
    device::triplet_counter_container_types::view triplet_view) {

    device::count_triplets(threadIdx.x + blockIdx.x * blockDim.x, config,
                           sp_grid, doublet_counter_view, doublet_prefix_sum,
                           mb_doublets, mt_doublets, triplet_view);
}
/// CUDA kernel for running @c traccc::device::find_triplets
__global__ void find_triplets(
    seedfinder_config config, seedfilter_config filter_config,
    sp_grid_const_view sp_grid,
    device::doublet_counter_container_types::const_view doublet_counter_view,
    doublet_container_view mb_doublets, doublet_container_view mt_doublets,
    device::triplet_counter_container_types::const_view tc_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        triplet_prefix_sum,
    triplet_container_view triplet_view) {

    device::find_triplets(threadIdx.x + blockIdx.x * blockDim.x, config,
                          filter_config, sp_grid, doublet_counter_view,
                          mb_doublets, mt_doublets, tc_view, triplet_prefix_sum,
                          triplet_view);
}
/// CUDA kernel for running @c traccc::device::update_triplet_weights
__global__ void update_triplet_weights(
    seedfilter_config filter_config, sp_grid_const_view sp_grid,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        triplet_prefix_sum,
    triplet_container_view triplet_view) {

    // Array for temporary storage of quality parameters for comparing triplets
    // within weight updating kernel
    extern __shared__ scalar data[];
    // Each thread uses compatSeedLimit elements of the array
    scalar* dataPos = &data[threadIdx.x * filter_config.compatSeedLimit];

    device::update_triplet_weights(threadIdx.x + blockIdx.x * blockDim.x,
                                   filter_config, sp_grid, triplet_prefix_sum,
                                   dataPos, triplet_view);
}

/// CUDA kernel for running @c traccc::device::select_seeds
__global__ void select_seeds(
    seedfilter_config filter_config,
    spacepoint_container_types::const_view spacepoints_view,
    sp_grid_const_view internal_sp_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t> dc_ps_view,
    device::doublet_counter_container_types::const_view
        doublet_counter_container,
    triplet_container_view tc_view, vecmem::data::vector_view<seed> seed_view) {

    // Array for temporary storage of triplets for comparing within seed
    // selecting kernel
    extern __shared__ triplet data2[];
    // Each thread uses max_triplets_per_spM elements of the array
    triplet* dataPos = &data2[threadIdx.x * filter_config.max_triplets_per_spM];

    device::select_seeds(threadIdx.x + blockIdx.x * blockDim.x, filter_config,
                         spacepoints_view, internal_sp_view, dc_ps_view,
                         doublet_counter_container, tc_view, dataPos,
                         seed_view);
}

}  // namespace kernels

vecmem::data::vector_buffer<device::prefix_sum_element_t> make_prefix_sum_buff(
    const std::vector<device::prefix_sum_size_t>& sizes, vecmem::copy& copy,
    const traccc::memory_resource& mr) {

    const device::prefix_sum_buffer_t make_sum_result =
        device::make_prefix_sum_buffer(sizes, copy, mr);
    const vecmem::data::vector_view<const device::prefix_sum_size_t>
        sizes_sum_view = make_sum_result.view;
    const unsigned int totalSize = make_sum_result.totalSize;

    // Create buffer and view objects for prefix sum vector
    vecmem::data::vector_buffer<device::prefix_sum_element_t> prefix_sum_buff(
        totalSize, mr.main);
    copy.setup(prefix_sum_buff);

    // Fill the prefix sum vector
    kernels::fill_prefix_sum<<<(sizes_sum_view.size() / 32) + 1, 32>>>(
        sizes_sum_view, prefix_sum_buff);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return prefix_sum_buff;
}

seed_finding::seed_finding(const seedfinder_config& config,
                           const traccc::memory_resource& mr)
    : m_seedfinder_config(config), m_mr(mr) {

    // Initialize m_copy ptr based on memory resources that were given
    if (mr.host) {
        m_copy = std::make_unique<vecmem::cuda::copy>();
    } else {
        m_copy = std::make_unique<vecmem::copy>();
    }
}

vecmem::data::vector_buffer<seed> seed_finding::operator()(
    const spacepoint_container_types::const_view& spacepoints_view,
    const sp_grid_const_view& g2_view) const {
    // Get the sizes from the grid view
    auto grid_sizes = m_copy->get_sizes(g2_view._data_view);

    return this->operator()(spacepoints_view, g2_view, grid_sizes);
}

vecmem::data::vector_buffer<seed> seed_finding::operator()(
    const spacepoint_container_types::buffer& spacepoints_buffer,
    const sp_grid_buffer& g2_buffer) const {
    // Get the sizes from the grid buffer
    auto grid_sizes = m_copy->get_sizes(g2_buffer._buffer);

    return this->operator()(spacepoints_buffer, g2_buffer, grid_sizes);
}

vecmem::data::vector_buffer<seed> seed_finding::operator()(
    const spacepoint_container_types::const_view& spacepoints_view,
    const sp_grid_const_view& g2_view,
    const std::vector<unsigned int>& grid_sizes) const {

    // Create prefix sum buffer and its view
    vecmem::data::vector_buffer sp_grid_prefix_sum_buff =
        make_prefix_sum_buff(grid_sizes, *m_copy, m_mr);

    // Set up the doublet counter buffer.
    device::doublet_counter_container_types::buffer doublet_counter_buffer =
        device::make_doublet_counter_buffer(grid_sizes, *m_copy, m_mr.main,
                                            m_mr.host);

    // Calculate the number of threads and thread blocks to run the doublet
    // counting kernel for.
    const unsigned int nDoubletCountThreads = WARP_SIZE * 2;
    const unsigned int nDoubletCountBlocks =
        sp_grid_prefix_sum_buff.size() / nDoubletCountThreads + 1;

    // Count the number of doublets that we need to produce.
    kernels::count_doublets<<<nDoubletCountBlocks, nDoubletCountThreads>>>(
        m_seedfinder_config, g2_view, sp_grid_prefix_sum_buff,
        doublet_counter_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Get the summary values per bin.
    vecmem::vector<device::doublet_counter_header> doublet_counts(
        m_mr.host ? m_mr.host : &(m_mr.main));
    (*m_copy)(doublet_counter_buffer.headers, doublet_counts);

    // Set up the doublet buffers.
    device::doublet_buffer_pair doublet_buffers = device::make_doublet_buffers(
        doublet_counter_buffer, *m_copy, m_mr.main, m_mr.host);

    // Create prefix sum buffer and its view
    vecmem::data::vector_buffer doublet_prefix_sum_buff = make_prefix_sum_buff(
        m_copy->get_sizes(doublet_counter_buffer.items), *m_copy, m_mr);

    // Calculate the number of threads and thread blocks to run the doublet
    // finding kernel for.
    const unsigned int nDoubletFindThreads = WARP_SIZE * 2;
    const unsigned int nDoubletFindBlocks =
        doublet_prefix_sum_buff.size() / nDoubletFindThreads + 1;

    // Find all of the spacepoint doublets.
    kernels::find_doublets<<<nDoubletFindBlocks, nDoubletFindThreads>>>(
        m_seedfinder_config, g2_view, doublet_counter_buffer,
        doublet_prefix_sum_buff, doublet_buffers.middleBottom,
        doublet_buffers.middleTop);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    std::vector<std::size_t> mb_buffer_sizes(doublet_counts.size());
    std::transform(
        doublet_counts.begin(), doublet_counts.end(), mb_buffer_sizes.begin(),
        [](const device::doublet_counter_header& dc) { return dc.m_nMidBot; });

    // Set up the triplet counter buffer and its view
    device::triplet_counter_container_types::buffer triplet_counter_buffer =
        device::make_triplet_counter_buffer(mb_buffer_sizes, *m_copy, m_mr.main,
                                            m_mr.host);

    // Create prefix sum buffer and its view
    vecmem::data::vector_buffer mb_prefix_sum_buff = make_prefix_sum_buff(
        m_copy->get_sizes(doublet_buffers.middleBottom.items), *m_copy, m_mr);

    // Calculate the number of threads and thread blocks to run the doublet
    // counting kernel for.
    const unsigned int nTripletCountThreads = WARP_SIZE * 2;
    const unsigned int nTripletCountBlocks =
        mb_prefix_sum_buff.size() / nTripletCountThreads + 1;

    // Count the number of triplets that we need to produce.
    kernels::count_triplets<<<nTripletCountBlocks, nTripletCountThreads>>>(
        m_seedfinder_config, g2_view, doublet_counter_buffer,
        mb_prefix_sum_buff, doublet_buffers.middleBottom,
        doublet_buffers.middleTop, triplet_counter_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Set up the triplet buffer.
    triplet_container_buffer triplet_buffer = device::make_triplet_buffer(
        triplet_counter_buffer, *m_copy, m_mr.main, m_mr.host);
    triplet_container_view triplet_view(triplet_buffer);

    // Create prefix sum buffer and its view
    vecmem::data::vector_buffer triplet_counter_prefix_sum_buff =
        make_prefix_sum_buff(m_copy->get_sizes(triplet_counter_buffer.items),
                             *m_copy, m_mr);

    // Calculate the number of threads and thread blocks to run the triplet
    // finding kernel for.
    const unsigned int nTripletFindThreads = WARP_SIZE * 2;
    const unsigned int nTripletFindBlocks =
        triplet_counter_prefix_sum_buff.size() / nTripletFindThreads + 1;

    // Find all of the spacepoint triplets.
    kernels::find_triplets<<<nTripletFindBlocks, nTripletFindThreads>>>(
        m_seedfinder_config, m_seedfilter_config, g2_view,
        doublet_counter_buffer, doublet_buffers.middleBottom,
        doublet_buffers.middleTop, triplet_counter_buffer,
        triplet_counter_prefix_sum_buff, triplet_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Create prefix sum buffer and its view
    vecmem::data::vector_buffer triplet_prefix_sum_buff = make_prefix_sum_buff(
        m_copy->get_sizes(triplet_buffer.items), *m_copy, m_mr);

    // Calculate the number of threads and thread blocks to run the weight
    // updating kernel for.
    const unsigned int nWeightUpdatingThreads = WARP_SIZE * 2;
    const unsigned int nWeightUpdatingBlocks =
        triplet_prefix_sum_buff.size() / nWeightUpdatingThreads + 1;

    // Update the weights of all spacepoint triplets.
    kernels::update_triplet_weights<<<
        nWeightUpdatingBlocks, nWeightUpdatingThreads,
        sizeof(scalar) * m_seedfilter_config.compatSeedLimit *
            nWeightUpdatingThreads>>>(m_seedfilter_config, g2_view,
                                      triplet_prefix_sum_buff, triplet_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Take header of the triplet counter container buffer into host
    vecmem::vector<device::triplet_counter_header> tcc_headers(
        m_mr.host ? m_mr.host : &(m_mr.main));
    (*m_copy)(triplet_counter_buffer.headers, tcc_headers);

    // Get the number of seeds (triplets)
    unsigned int n_triplets = 0;
    for (const auto& h : tcc_headers) {
        n_triplets += h.m_nTriplets;
    }

    vecmem::data::vector_buffer<seed> seed_buffer(n_triplets, 0, m_mr.main);
    m_copy->setup(seed_buffer);

    // Calculate the number of threads and thread blocks to run the seed
    // selecting kernel for.
    const unsigned int nSeedSelectingThreads = WARP_SIZE * 2;
    const unsigned int nSeedSelectingBlocks =
        doublet_prefix_sum_buff.size() / nSeedSelectingThreads + 1;

    // Create seeds out of selected triplets
    kernels::select_seeds<<<nSeedSelectingBlocks, nSeedSelectingThreads,
                            sizeof(triplet) *
                                m_seedfilter_config.max_triplets_per_spM *
                                nSeedSelectingThreads>>>(
        m_seedfilter_config, spacepoints_view, g2_view, doublet_prefix_sum_buff,
        doublet_counter_buffer, triplet_buffer, seed_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    return seed_buffer;
}

}  // namespace traccc::cuda
