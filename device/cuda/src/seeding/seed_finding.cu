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
#include "traccc/seeding/detail/singlet.hpp"
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
namespace detail {
/**
 * @brief Encode the state of our parameter insertion mutex.
 */
TRACCC_HOST_DEVICE inline uint64_t encode_insertion_mutex(const bool locked,
                                                          const uint32_t size,
                                                          const float max) {

    // Assert that the MSB of the size is zero
    assert(size <= 0x7FFFFFFF);
    const uint32_t hi = size | (locked ? 0x80000000 : 0x0);
    const uint32_t lo = std::bit_cast<uint32_t>(max);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

/**
 * @brief Decode the state of our parameter insertion mutex.
 */
TRACCC_HOST_DEVICE inline std::tuple<bool, uint32_t, float>
decode_insertion_mutex(const uint64_t val) {
    const uint32_t hi = static_cast<uint32_t>(val >> 32);
    const uint32_t lo = val & 0xFFFFFFFF;
    return {static_cast<bool>(hi & 0x80000000), (hi & 0x7FFFFFFF),
            std::bit_cast<float>(lo)};
}
}  // namespace detail

__global__ inline void gather_best_triplets_per_spacepoint(
    const device::device_triplet_collection_types::const_view triplet_view,
    const traccc::details::spacepoint_grid_types::const_view grid_view,
    vecmem::data::vector_view<unsigned long long int> insertion_mutex_view,
    vecmem::data::vector_view<unsigned int> triplet_index_view,
    vecmem::data::vector_view<float> triplet_weight_view,
    const unsigned int max_num_triplets_per_spacepoint) {

    const device::device_triplet_collection_types::const_device triplets(
        triplet_view);
    const traccc::details::spacepoint_grid_types::const_device grid(grid_view);

    vecmem::device_vector<unsigned long long int> insertion_mutex(
        insertion_mutex_view);
    vecmem::device_vector<unsigned int> triplet_index(triplet_index_view);
    vecmem::device_vector<float> triplet_weight(triplet_weight_view);

    unsigned int triplet_idx = blockIdx.x * blockDim.x + threadIdx.x;

    scalar weight;
    bool need_to_write = true;
    unsigned int current_state = 0;
    unsigned int spacepoint_idx = 0;

    auto get_idx_for_state = [&triplets, &grid,
                              triplet_idx](unsigned int state) {
        const device::device_triplet& this_triplet = triplets.at(triplet_idx);
        if (state == 0) {
            return this_triplet.spB;
        } else if (state == 1) {
            return this_triplet.spM;
        } else {
            return this_triplet.spT;
        }
    };

    if (triplet_idx < triplets.size()) {
        weight = triplets.at(triplet_idx).weight;
        spacepoint_idx = get_idx_for_state(0);
    } else {
        need_to_write = false;
        current_state = 3;
    }

    while (__syncthreads_or(current_state < 3 || need_to_write)) {
        if (current_state < 3 || need_to_write) {
            if (need_to_write) {
                vecmem::device_atomic_ref<unsigned long long int> mutex(
                    insertion_mutex.at(spacepoint_idx));

                unsigned long long int assumed = mutex.load();
                unsigned long long int desired_set;

                auto [locked, size, worst] =
                    detail::decode_insertion_mutex(assumed);

                if (need_to_write && size >= max_num_triplets_per_spacepoint &&
                    weight <= worst) {
                    need_to_write = false;
                }

                bool holds_lock = false;

                if (need_to_write && !locked) {
                    desired_set =
                        detail::encode_insertion_mutex(true, size, worst);
                    if (mutex.compare_exchange_strong(assumed, desired_set)) {
                        holds_lock = true;
                    }
                }

                if (holds_lock) {
                    unsigned int new_size;
                    unsigned int offset =
                        spacepoint_idx * max_num_triplets_per_spacepoint;
                    unsigned int out_idx;

                    if (size == max_num_triplets_per_spacepoint) {
                        new_size = size;
                        scalar worst_weight =
                            std::numeric_limits<scalar>::max();
                        for (unsigned int i = 0; i < size; ++i) {
                            if (triplet_weight.at(offset + i) < worst_weight) {
                                worst_weight = triplet_weight.at(offset + i);
                                out_idx = i;
                            }
                        }
                    } else {
                        new_size = size + 1;
                        out_idx = size;
                    }

                    triplet_index.at(offset + out_idx) = triplet_idx;
                    triplet_weight.at(offset + out_idx) = weight;

                    scalar new_worst = std::numeric_limits<scalar>::max();

                    for (unsigned int i = 0; i < new_size; ++i) {
                        new_worst =
                            std::min(new_worst, triplet_weight.at(offset + i));
                    }

                    [[maybe_unused]] bool cas_result =
                        mutex.compare_exchange_strong(
                            desired_set, detail::encode_insertion_mutex(
                                             false, new_size, new_worst));
                    assert(cas_result);

                    need_to_write = false;
                }
            }

            if (!need_to_write && current_state < 3) {
                if (current_state < 2) {
                    spacepoint_idx = get_idx_for_state(++current_state);
                    need_to_write = true;
                } else {
                    ++current_state;
                }
            }
        }
    }
}

__global__ inline void gather_spacepoint_votes(
    const vecmem::data::vector_view<const unsigned long long int>
        insertion_mutex_view,
    const vecmem::data::vector_view<const unsigned int> triplet_index_view,
    vecmem::data::vector_view<unsigned int> votes_per_triplet_view,
    const unsigned int max_num_triplets_per_spacepoint) {

    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int spacepoint_idx = thread_idx / max_num_triplets_per_spacepoint;
    unsigned int triplet_idx = thread_idx % max_num_triplets_per_spacepoint;

    const vecmem::device_vector<const unsigned long long int> insertion_mutex(
        insertion_mutex_view);
    const vecmem::device_vector<const unsigned int> triplet_index(
        triplet_index_view);
    vecmem::device_vector<unsigned int> votes_per_triplet(
        votes_per_triplet_view);

    if (spacepoint_idx >= insertion_mutex.size()) {
        return;
    }

    auto [locked, size, worst] =
        detail::decode_insertion_mutex(insertion_mutex.at(spacepoint_idx));

    if (size == 0) {
        return;
    }

    unsigned int num_votes =
        (size + max_num_triplets_per_spacepoint - 1) / size;

    if (triplet_idx < size) {
        vecmem::device_atomic_ref<unsigned int>(
            votes_per_triplet.at(triplet_index.at(thread_idx)))
            .fetch_add(num_votes);
    }
}

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
    const vecmem::data::vector_view<const unsigned int> votes_per_triplet_view,
    edm::seed_collection::view seed_view) {

    // Array for temporary storage of triplets for comparing within seed
    // selecting kernel
    extern __shared__ device::device_triplet data2[];
    // Each thread uses max_triplets_per_spM elements of the array
    device::device_triplet* dataPos =
        &data2[threadIdx.x * finder_config.maxSeedsPerSpM];

    device::select_seeds(details::global_index1(), finder_config, filter_config,
                         spacepoints, sp_view, spM_tc, midBot_tc, triplet_view,
                         dataPos, votes_per_triplet_view, seed_view);
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

    // Pointer to stage device-to-host copies for container sizes
    vecmem::unique_alloc_ptr<unsigned int> size_staging_ptr =
        vecmem::make_unique_alloc<unsigned int>(*(m_mr.host));

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

    // Number of spacepoints including those not in the grid.
    const unsigned int num_spacepoints_total =
        m_copy.get_size(spacepoints_view);

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

    // Transfer the doublet count to the host.
    TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
        size_staging_ptr.get(), doublet_counter_buffer.size_ptr(),
        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

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
    const unsigned int doublet_counter_buffer_size = *size_staging_ptr;
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
    TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
        size_staging_ptr.get(), triplet_counter_midBot_buffer.size_ptr(),
        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
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
        (*size_staging_ptr + nTripletFindThreads - 1) / nTripletFindThreads;

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

    vecmem::data::vector_buffer<unsigned int> votes_per_triplet_buffer(
        globalCounter_host->m_nTriplets, m_mr.main);
    m_copy.setup(votes_per_triplet_buffer)->wait();
    m_copy.memset(votes_per_triplet_buffer, 0)->wait();

    {
        unsigned int num_votes_per_sp = 3;

        vecmem::data::vector_buffer<unsigned int>
            best_triplets_per_spacepoint_index_buffer(
                num_votes_per_sp * num_spacepoints_total, m_mr.main);
        m_copy.setup(best_triplets_per_spacepoint_index_buffer)->wait();

        vecmem::data::vector_buffer<unsigned long long int>
            best_triplets_per_spacepoint_insertion_mutex_buffer(
                num_spacepoints_total, m_mr.main);
        m_copy.setup(best_triplets_per_spacepoint_insertion_mutex_buffer)
            ->wait();
        m_copy.memset(best_triplets_per_spacepoint_insertion_mutex_buffer, 0)
            ->wait();

        {
            vecmem::data::vector_buffer<float>
                best_triplets_per_spacepoint_weight_buffer(
                    num_votes_per_sp * num_spacepoints_total, m_mr.main);
            m_copy.setup(best_triplets_per_spacepoint_weight_buffer)->wait();

            unsigned int num_threads = 64;
            unsigned int num_blocks =
                (globalCounter_host->m_nTriplets + num_threads - 1) /
                num_threads;

            kernels::gather_best_triplets_per_spacepoint<<<
                num_blocks, num_threads, 0, stream>>>(
                triplet_buffer, g2_view,
                best_triplets_per_spacepoint_insertion_mutex_buffer,
                best_triplets_per_spacepoint_index_buffer,
                best_triplets_per_spacepoint_weight_buffer, num_votes_per_sp);

            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }

        {
            unsigned int num_threads = 512;
            unsigned int num_blocks =
                (num_votes_per_sp * num_spacepoints_total + num_threads - 1) /
                num_threads;

            kernels::
                gather_spacepoint_votes<<<num_blocks, num_threads, 0, stream>>>(
                    best_triplets_per_spacepoint_insertion_mutex_buffer,
                    best_triplets_per_spacepoint_index_buffer,
                    votes_per_triplet_buffer, num_votes_per_sp);

            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
        }
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
        triplet_buffer, votes_per_triplet_buffer, seed_buffer);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    return seed_buffer;
}

}  // namespace details
}  // namespace traccc::cuda
