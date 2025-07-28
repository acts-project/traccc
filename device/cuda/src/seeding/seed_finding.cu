/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/seeding/details/seed_finding.hpp"

// Project include(s).
#include "traccc/cuda/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/device/make_prefix_sum_buffer.hpp"
#include "traccc/edm/device/device_doublet.hpp"
#include "traccc/edm/device/device_triplet.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/edm/device/seeding_global_counter.hpp"
#include "traccc/edm/device/triplet_counter.hpp"
#include "traccc/seeding/device/count_doublets.hpp"
#include "traccc/seeding/device/count_triplets.hpp"
#include "traccc/seeding/device/find_doublets.hpp"
#include "traccc/seeding/device/find_triplets.hpp"
#include "traccc/seeding/device/reduce_triplet_counts.hpp"
#include "traccc/seeding/device/select_seeds.hpp"
#include "traccc/seeding/device/update_triplet_weights.hpp"

// VecMem include(s).
#include <vecmem/utils/cuda/copy.hpp>

// System include(s).
#include <algorithm>
#include <vector>

namespace traccc::cuda {
namespace kernels {

/// CUDA kernel for running @c traccc::device::count_doublets
__global__ void count_doublets(
    seedfinder_config config,
    edm::spacepoint_collection::const_view spacepoints,
    traccc::details::spacepoint_grid_types::const_view sp_grid,
    vecmem::data::vector_view<const device::prefix_sum_element_t> sp_prefix_sum,
    device::doublet_counter_collection_types::view doublet_counter,
    unsigned int& nMidBot, unsigned int& nMidTop) {

    device::count_doublets(details::global_index1(), config, spacepoints,
                           sp_grid, sp_prefix_sum, doublet_counter, nMidBot,
                           nMidTop);
}

/// CUDA kernel for running @c traccc::device::find_doublets
__global__ void find_doublets(
    seedfinder_config config,
    edm::spacepoint_collection::const_view spacepoints,
    traccc::details::spacepoint_grid_types::const_view sp_grid,
    device::doublet_counter_collection_types::const_view doublet_counter,
    device::device_doublet_collection_types::view mb_doublets,
    device::device_doublet_collection_types::view mt_doublets) {

    device::find_doublets(details::global_index1(), config, spacepoints,
                          sp_grid, doublet_counter, mb_doublets, mt_doublets);
}

/// CUDA kernel for running @c traccc::device::count_triplets
__global__ void count_triplets(
    seedfinder_config config,
    edm::spacepoint_collection::const_view spacepoints,
    traccc::details::spacepoint_grid_types::const_view sp_grid,
    device::doublet_counter_collection_types::const_view doublet_counter,
    device::device_doublet_collection_types::const_view mb_doublets,
    device::device_doublet_collection_types::const_view mt_doublets,
    device::triplet_counter_spM_collection_types::view spM_counter,
    device::triplet_counter_collection_types::view midBot_counter) {

    device::count_triplets(details::global_index1(), config, spacepoints,
                           sp_grid, doublet_counter, mb_doublets, mt_doublets,
                           spM_counter, midBot_counter);
}

/// CUDA kernel for running @c traccc::device::reduce_triplet_counts
__global__ void reduce_triplet_counts(
    device::doublet_counter_collection_types::const_view doublet_counter,
    device::triplet_counter_spM_collection_types::view spM_counter,
    unsigned int& num_triplets) {

    device::reduce_triplet_counts(details::global_index1(), doublet_counter,
                                  spM_counter, num_triplets);
}

/// CUDA kernel for running @c traccc::device::find_triplets
__global__ void find_triplets(
    seedfinder_config config, seedfilter_config filter_config,
    edm::spacepoint_collection::const_view spacepoints,
    traccc::details::spacepoint_grid_types::const_view sp_grid,
    device::doublet_counter_collection_types::const_view doublet_counter,
    device::device_doublet_collection_types::const_view mt_doublets,
    device::triplet_counter_spM_collection_types::const_view spM_tc,
    device::triplet_counter_collection_types::const_view midBot_tc,
    device::device_triplet_collection_types::view triplet_view) {

    device::find_triplets(details::global_index1(), config, filter_config,
                          spacepoints, sp_grid, doublet_counter, mt_doublets,
                          spM_tc, midBot_tc, triplet_view);
}

/// CUDA kernel for running @c traccc::device::update_triplet_weights
__global__ void update_triplet_weights(
    seedfilter_config filter_config,
    edm::spacepoint_collection::const_view spacepoints,
    device::triplet_counter_spM_collection_types::const_view spM_tc,
    device::triplet_counter_collection_types::const_view midBot_tc,
    device::device_triplet_collection_types::view triplet_view) {

    // Array for temporary storage of quality parameters for comparing triplets
    // within weight updating kernel
    extern __shared__ scalar data[];
    // Each thread uses compatSeedLimit elements of the array
    scalar* dataPos = &data[threadIdx.x * filter_config.compatSeedLimit];

    device::update_triplet_weights(details::global_index1(), filter_config,
                                   spacepoints, spM_tc, midBot_tc, dataPos,
                                   triplet_view);
}

/// CUDA kernel for running @c traccc::device::select_seeds
__global__ void select_seeds(
    seedfinder_config finder_config, seedfilter_config filter_config,
    edm::spacepoint_collection::const_view spacepoints,
    traccc::details::spacepoint_grid_types::const_view sp_view,
    device::triplet_counter_spM_collection_types::const_view spM_tc,
    device::triplet_counter_collection_types::const_view midBot_tc,
    device::device_triplet_collection_types::view triplet_view,
    vecmem::data::vector_view<const scalar> highest_weight_view,
    edm::seed_collection::view seed_view) {

    // Array for temporary storage of triplets for comparing within seed
    // selecting kernel
    extern __shared__ device::device_triplet data2[];
    // Each thread uses max_triplets_per_spM elements of the array
    device::device_triplet* dataPos =
        &data2[threadIdx.x * finder_config.maxSeedsPerSpM];

    device::select_seeds(details::global_index1(), finder_config, filter_config,
                         spacepoints, sp_view, spM_tc, midBot_tc, triplet_view,
                         highest_weight_view, dataPos, seed_view);
}

__global__ void collect_highest_weight_per_spacepoint(
    vecmem::data::vector_view<scalar> highest_weight_view,
    device::device_triplet_collection_types::const_view triplet_view) {
    vecmem::device_vector<scalar> highest_weights(highest_weight_view);
    const device::device_triplet_collection_types::const_device triplets(
        triplet_view);

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int seed_idx = tid / 3;
    const unsigned int local_spacepoint_idx = tid % 3;

    if (seed_idx >= triplets.size()) {
        return;
    }

    unsigned int global_spacepoint_idx;

    if (local_spacepoint_idx == 0) {
        global_spacepoint_idx = triplets.at(seed_idx).spB;
    } else if (local_spacepoint_idx == 1) {
        global_spacepoint_idx = triplets.at(seed_idx).spM;
    } else if (local_spacepoint_idx == 2) {
        global_spacepoint_idx = triplets.at(seed_idx).spT;
    } else {
        __builtin_unreachable();
    }

    static_assert(sizeof(scalar) == 4 || sizeof(scalar) == 8);
    using cas_type = std::conditional_t<sizeof(scalar) == 4, unsigned int,
                                        unsigned long long int>;

    /*
     * The following is simply an implementation of atomic max.
     */
    scalar& weight_loc = highest_weights.at(global_spacepoint_idx);
    cas_type* weight_raw = reinterpret_cast<cas_type*>(&weight_loc);

    vecmem::device_atomic_ref<cas_type> atomic(*weight_raw);

    cas_type current_weight_raw = atomic.load();
    scalar current_weight = std::bit_cast<scalar>(current_weight_raw);
    const scalar own_weight = triplets.at(seed_idx).weight;
    const cas_type own_weight_raw = std::bit_cast<cas_type>(own_weight);

    while (own_weight > current_weight) {
        const bool res =
            atomic.compare_exchange_strong(current_weight_raw, own_weight_raw);

        if (res) {
            return;
        } else {
            current_weight = std::bit_cast<scalar>(current_weight_raw);
        }
    }
}
}  // namespace kernels

namespace details {

seed_finding::seed_finding(const seedfinder_config& config,
                           const seedfilter_config& filter_config,
                           const traccc::memory_resource& mr,
                           vecmem::copy& copy, stream& str,
                           std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_seedfinder_config(config),
      m_seedfilter_config(filter_config),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

edm::seed_collection::buffer seed_finding::operator()(
    const edm::spacepoint_collection::const_view& spacepoints_view,
    const traccc::details::spacepoint_grid_types::const_view& g2_view) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Get the sizes from the grid view
    auto grid_sizes = m_copy.get_sizes(g2_view._data_view);

    // Create prefix sum buffer
    vecmem::data::vector_buffer sp_grid_prefix_sum_buff =
        make_prefix_sum_buff(grid_sizes, m_copy, m_mr, m_stream);

    const auto num_spacepoints = m_copy.get_size(sp_grid_prefix_sum_buff);
    if (num_spacepoints == 0) {
        return {0, m_mr.main};
    }

    // Total number of spacepoints, including those not in the grid.
    const auto total_num_spacepoints = m_copy.get_size(spacepoints_view);

    // Set up the doublet counter buffer.
    device::doublet_counter_collection_types::buffer doublet_counter_buffer = {
        num_spacepoints, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.setup(doublet_counter_buffer)->ignore();

    // Calculate the number of threads and thread blocks to run the doublet
    // counting kernel for.
    const unsigned int nDoubletCountThreads = m_warp_size * 2;
    const unsigned int nDoubletCountBlocks =
        (num_spacepoints + nDoubletCountThreads - 1) / nDoubletCountThreads;

    // Counter for the total number of doublets and triplets
    vecmem::unique_alloc_ptr<device::seeding_global_counter>
        globalCounter_device =
            vecmem::make_unique_alloc<device::seeding_global_counter>(
                m_mr.main);
    TRACCC_CUDA_ERROR_CHECK(
        cudaMemsetAsync(globalCounter_device.get(), 0,
                        sizeof(device::seeding_global_counter), stream));

    // Count the number of doublets that we need to produce.
    kernels::count_doublets<<<nDoubletCountBlocks, nDoubletCountThreads, 0,
                              stream>>>(
        m_seedfinder_config, spacepoints_view, g2_view, sp_grid_prefix_sum_buff,
        doublet_counter_buffer, (*globalCounter_device).m_nMidBot,
        (*globalCounter_device).m_nMidTop);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Get the summary values.
    vecmem::unique_alloc_ptr<device::seeding_global_counter>
        globalCounter_host =
            vecmem::make_unique_alloc<device::seeding_global_counter>(
                (m_mr.host != nullptr) ? *(m_mr.host) : m_mr.main);
    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(globalCounter_host.get(), globalCounter_device.get(),
                        sizeof(device::seeding_global_counter),
                        cudaMemcpyDeviceToHost, stream));
    m_stream.synchronize();

    if (globalCounter_host->m_nMidBot == 0 ||
        globalCounter_host->m_nMidTop == 0) {
        return {0, m_mr.main};
    }

    // Set up the doublet counter buffers.
    device::device_doublet_collection_types::buffer doublet_buffer_mb = {
        globalCounter_host->m_nMidBot, m_mr.main};
    m_copy.setup(doublet_buffer_mb)->ignore();
    device::device_doublet_collection_types::buffer doublet_buffer_mt = {
        globalCounter_host->m_nMidTop, m_mr.main};
    m_copy.setup(doublet_buffer_mt)->ignore();

    // Calculate the number of threads and thread blocks to run the doublet
    // finding kernel for.
    const unsigned int nDoubletFindThreads = m_warp_size * 2;
    const unsigned int doublet_counter_buffer_size =
        m_copy.get_size(doublet_counter_buffer);
    const unsigned int nDoubletFindBlocks =
        (doublet_counter_buffer_size + nDoubletFindThreads - 1) /
        nDoubletFindThreads;

    // Find all of the spacepoint doublets.
    kernels::
        find_doublets<<<nDoubletFindBlocks, nDoubletFindThreads, 0, stream>>>(
            m_seedfinder_config, spacepoints_view, g2_view,
            doublet_counter_buffer, doublet_buffer_mb, doublet_buffer_mt);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Set up the triplet counter buffers
    device::triplet_counter_spM_collection_types::buffer
        triplet_counter_spM_buffer = {doublet_counter_buffer_size, m_mr.main};
    m_copy.setup(triplet_counter_spM_buffer)->ignore();
    m_copy.memset(triplet_counter_spM_buffer, 0)->ignore();
    device::triplet_counter_collection_types::buffer
        triplet_counter_midBot_buffer = {globalCounter_host->m_nMidBot,
                                         m_mr.main,
                                         vecmem::data::buffer_type::resizable};
    m_copy.setup(triplet_counter_midBot_buffer)->ignore();

    // Calculate the number of threads and thread blocks to run the doublet
    // counting kernel for.
    const unsigned int nTripletCountThreads = m_warp_size * 2;
    const unsigned int nTripletCountBlocks =
        (globalCounter_host->m_nMidBot + nTripletCountThreads - 1) /
        nTripletCountThreads;

    // Count the number of triplets that we need to produce.
    kernels::count_triplets<<<nTripletCountBlocks, nTripletCountThreads, 0,
                              stream>>>(
        m_seedfinder_config, spacepoints_view, g2_view, doublet_counter_buffer,
        doublet_buffer_mb, doublet_buffer_mt, triplet_counter_spM_buffer,
        triplet_counter_midBot_buffer);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Calculate the number of threads and thread blocks to run the triplet
    // count reduction kernel for.
    const unsigned int nTcReductionThreads = m_warp_size * 2;
    const unsigned int nTcReductionBlocks =
        (doublet_counter_buffer_size + nTcReductionThreads - 1) /
        nTcReductionThreads;

    // Reduce the triplet counts per spM.
    kernels::reduce_triplet_counts<<<nTcReductionBlocks, nTcReductionThreads, 0,
                                     stream>>>(
        doublet_counter_buffer, triplet_counter_spM_buffer,
        (*globalCounter_device).m_nTriplets);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    TRACCC_CUDA_ERROR_CHECK(
        cudaMemcpyAsync(globalCounter_host.get(), globalCounter_device.get(),
                        sizeof(device::seeding_global_counter),
                        cudaMemcpyDeviceToHost, stream));
    m_stream.synchronize();

    if (globalCounter_host->m_nTriplets == 0) {
        return {0, m_mr.main};
    }

    // Set up the triplet buffer.
    device::device_triplet_collection_types::buffer triplet_buffer = {
        globalCounter_host->m_nTriplets, m_mr.main};
    m_copy.setup(triplet_buffer)->ignore();

    // Calculate the number of threads and thread blocks to run the triplet
    // finding kernel for.
    const unsigned int nTripletFindThreads = m_warp_size * 2;
    const unsigned int nTripletFindBlocks =
        (m_copy.get_size(triplet_counter_midBot_buffer) + nTripletFindThreads -
         1) /
        nTripletFindThreads;

    // Find all of the spacepoint triplets.
    kernels::
        find_triplets<<<nTripletFindBlocks, nTripletFindThreads, 0, stream>>>(
            m_seedfinder_config, m_seedfilter_config, spacepoints_view, g2_view,
            doublet_counter_buffer, doublet_buffer_mt,
            triplet_counter_spM_buffer, triplet_counter_midBot_buffer,
            triplet_buffer);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Calculate the number of threads and thread blocks to run the weight
    // updating kernel for.
    const unsigned int nWeightUpdatingThreads = m_warp_size * 2;
    const unsigned int nWeightUpdatingBlocks =
        (globalCounter_host->m_nTriplets + nWeightUpdatingThreads - 1) /
        nWeightUpdatingThreads;

    // Update the weights of all spacepoint triplets.
    kernels::update_triplet_weights<<<
        nWeightUpdatingBlocks, nWeightUpdatingThreads,
        sizeof(scalar) * m_seedfilter_config.compatSeedLimit *
            nWeightUpdatingThreads,
        stream>>>(m_seedfilter_config, spacepoints_view,
                  triplet_counter_spM_buffer, triplet_counter_midBot_buffer,
                  triplet_buffer);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    TRACCC_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

    vecmem::data::vector_buffer<scalar> highest_weight_buffer(0, m_mr.main);

    if (m_seedfilter_config.minNumTimesWeightCompatible > 0) {
        highest_weight_buffer = {total_num_spacepoints, m_mr.main};

        /*
         * NOTE: The value 0b11100000 is chosen because it, when repeated four
         * times to create a 32-bit floating point value, evaluates to the
         * number 0b11100000111000001110000011100000 which is very small, much
         * smaller than any seed weight should ever be. When broadcast eight
         * times to a 64-bit number it also produces an incredibly small value.
         */
        m_copy.memset(highest_weight_buffer, 0b11100000)->wait();

        const unsigned int num_votes = 3 * globalCounter_host->m_nTriplets;
        const unsigned int num_threads = 256;
        const unsigned int num_blocks =
            (num_votes + num_threads - 1) / num_threads;

        kernels::collect_highest_weight_per_spacepoint<<<
            num_blocks, num_threads, 0, stream>>>(highest_weight_buffer,
                                                  triplet_buffer);
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
    }

    // Create result object: collection of seeds
    edm::seed_collection::buffer seed_buffer(
        globalCounter_host->m_nTriplets, m_mr.main,
        vecmem::data::buffer_type::resizable);
    m_copy.setup(seed_buffer)->ignore();

    // Calculate the number of threads and thread blocks to run the seed
    // selecting kernel for.
    const unsigned int nSeedSelectingThreads = m_warp_size * 2;
    const unsigned int nSeedSelectingBlocks =
        (doublet_counter_buffer_size + nSeedSelectingThreads - 1) /
        nSeedSelectingThreads;

    // Create seeds out of selected triplets
    kernels::select_seeds<<<nSeedSelectingBlocks, nSeedSelectingThreads,
                            sizeof(device::device_triplet) *
                                m_seedfinder_config.maxSeedsPerSpM *
                                nSeedSelectingThreads,
                            stream>>>(
        m_seedfinder_config, m_seedfilter_config, spacepoints_view, g2_view,
        triplet_counter_spM_buffer, triplet_counter_midBot_buffer,
        triplet_buffer, highest_weight_buffer, seed_buffer);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    return seed_buffer;
}

}  // namespace details
}  // namespace traccc::cuda
