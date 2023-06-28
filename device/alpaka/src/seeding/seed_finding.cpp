/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/seeding/seed_finding.hpp"
#include "traccc/alpaka/utils/definitions.hpp"

// Project include(s).
#include "traccc/alpaka/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/device/make_prefix_sum_buffer.hpp"
#include "traccc/edm/device/device_doublet.hpp"
#include "traccc/edm/device/device_triplet.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
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
#include "vecmem/utils/cuda/copy.hpp"

// System include(s).
#include <algorithm>
#include <vector>

namespace traccc::alpaka {
// namespace kernels {

/// Kernel for running @c traccc::device::count_doublets
struct CountDoubletsKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc,
        seedfinder_config config,
        sp_grid_const_view sp_grid,
        vecmem::data::vector_view<const device::prefix_sum_element_t> sp_prefix_sum,
        device::doublet_counter_collection_types::view doublet_counter,
        unsigned int& nMidBot, unsigned int& nMidTop
    ) const
    {
        auto const globalThreadIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
        device::count_doublets(globalThreadIdx, config, sp_grid, sp_prefix_sum, doublet_counter, nMidBot, nMidTop);
    }
};

// Kernel for running @c traccc::device::find_doublets
struct FindDoubletsKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc,
        seedfinder_config config, sp_grid_const_view sp_grid,
        device::doublet_counter_collection_types::const_view doublet_counter,
        device::device_doublet_collection_types::view mb_doublets,
        device::device_doublet_collection_types::view mt_doublets
    ) const
    {
        auto const globalThreadIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
        device::find_doublets(globalThreadIdx, config, sp_grid, doublet_counter, mb_doublets, mt_doublets);
    }
};

// Kernel for running @c traccc::device::count_triplets
struct CountTripletsKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc,
        seedfinder_config config, sp_grid_const_view sp_grid,
        device::doublet_counter_collection_types::const_view doublet_counter,
        device::device_doublet_collection_types::const_view mb_doublets,
        device::device_doublet_collection_types::const_view mt_doublets,
        device::triplet_counter_spM_collection_types::view spM_counter,
        device::triplet_counter_collection_types::view midBot_counter
    ) const
    {
        auto const globalThreadIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
        device::count_triplets(globalThreadIdx, config, sp_grid,
                               doublet_counter, mb_doublets, mt_doublets,
                               spM_counter, midBot_counter);
    }
};

// Kernel for running @c traccc::device::reduce_triplet_counts
struct ReduceTripletCounts {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc,
        device::doublet_counter_collection_types::const_view doublet_counter,
        device::triplet_counter_spM_collection_types::view spM_counter,
        unsigned int& num_triplets
    ) const
    {
        auto const globalThreadIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
        device::reduce_triplet_counts(globalThreadIdx, doublet_counter, spM_counter, num_triplets);
    }
};

// Kernel for running @c traccc::device::find_triplets
struct FindTripletsKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc,
        seedfinder_config config, seedfilter_config filter_config,
        sp_grid_const_view sp_grid,
        device::doublet_counter_collection_types::const_view doublet_counter,
        device::device_doublet_collection_types::const_view mt_doublets,
        device::triplet_counter_spM_collection_types::const_view spM_tc,
        device::triplet_counter_collection_types::const_view midBot_tc,
        device::device_triplet_collection_types::view triplet_view
    ) const
    {

        auto const globalThreadIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];
        device::find_triplets(globalThreadIdx, config,
                              filter_config, sp_grid, doublet_counter, mt_doublets,
                              spM_tc, midBot_tc, triplet_view);
    }
};

// Kernel for running @c traccc::device::update_triplet_weights
struct UpdateTripletWeightsKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc,
        seedfilter_config filter_config, sp_grid_const_view sp_grid,
        device::triplet_counter_spM_collection_types::const_view spM_tc,
        device::triplet_counter_collection_types::const_view midBot_tc,
        device::device_triplet_collection_types::view triplet_view
    ) const
    {
        auto const globalThreadIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];

        // TODO: Fix use of shared.

        // // Array for temporary storage of quality parameters for comparing triplets
        // // within weight updating kernel
        // extern __shared__ scalar data[];
        // // Each thread uses compatSeedLimit elements of the array
        // scalar* dataPos = &data[threadIdx.x * filter_config.compatSeedLimit];

        // device::update_triplet_weights(threadIdx.x + blockIdx.x * blockDim.x,
        //                                filter_config, sp_grid, triplet_prefix_sum,
        //                                dataPos, triplet_view);
    }
};

// Kernel for running @c traccc::device::select_seeds
struct SelectSeedsKernel {
    template <typename Acc>
    ALPAKA_FN_ACC void operator()(
        Acc const& acc,
        seedfilter_config filter_config,
        spacepoint_collection_types::const_view spacepoints_view,
        sp_grid_const_view internal_sp_view,
        device::triplet_counter_spM_collection_types::const_view spM_tc,
        device::triplet_counter_collection_types::const_view midBot_tc,
        device::device_triplet_collection_types::view triplet_view,
        seed_collection_types::view seed_view
    ) const
    {
        auto const globalThreadIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u];

        // TODO: Fix use of shared.

        // // Array for temporary storage of triplets for comparing within seed
        // // selecting kernel
        // extern __shared__ triplet data2[];
        // // Each thread uses max_triplets_per_spM elements of the array
        // triplet* dataPos = &data2[threadIdx.x * filter_config.max_triplets_per_spM];

        // device::select_seeds(threadIdx.x + blockIdx.x * blockDim.x, filter_config,
        //                      spacepoints_view, internal_sp_view, dc_ps_view,
        //                      doublet_counter_container, tc_view, dataPos,
        //                      seed_view);
    }

};

seed_finding::seed_finding(const seedfinder_config& config,
                           const seedfilter_config& filter_config,
                           const traccc::memory_resource& mr,
                           vecmem::copy& copy)
    : m_seedfinder_config(config),
      m_seedfilter_config(filter_config),
      m_mr(mr),
      m_copy(copy) {}

seed_finding::output_type seed_finding::operator()(
    const spacepoint_collection_types::const_view& spacepoints_view,
    const sp_grid_const_view& g2_view) const {

    // Setup alpaka
    auto devAcc = ::alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};
    auto const deviceProperties = ::alpaka::getAccDevProps<Acc>(devAcc);
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];
    auto const threadsPerBlock = maxThreadsPerBlock;

    // Get the sizes from the grid view
    auto grid_sizes = m_copy.get_sizes(g2_view._data_view);

    // Create prefix sum buffer
    vecmem::data::vector_buffer sp_grid_prefix_sum_buff =
        make_prefix_sum_buff(
            grid_sizes, m_copy, m_mr, queue
        );

    const auto num_spacepoints = m_copy.get_size(sp_grid_prefix_sum_buff);

    if (num_spacepoints == 0) {
        return {0, m_mr.main};
    }

    // Set up the doublet counter buffer.
    device::doublet_counter_collection_types::buffer doublet_counter_buffer = {
        m_copy.get_size(sp_grid_prefix_sum_buff), m_mr.main,
        vecmem::data::buffer_type::resizable};
    m_copy.setup(doublet_counter_buffer);

    // Calculate the number of threads and thread blocks to run the doublet
    // counting kernel for.
    auto blocksPerGrid = (sp_grid_prefix_sum_buff.size() + threadsPerBlock - 1) / threadsPerBlock;
    auto elementsPerThread = 1u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "Doublet counter buffer of size" << sp_grid_prefix_sum_buff.size() << std::endl;

    // Counter for the total number of doublets and triplets
    vecmem::unique_alloc_ptr<device::seeding_global_counter>
        globalCounter_device =
            vecmem::make_unique_alloc<device::seeding_global_counter>(
                m_mr.main);
    auto globalCounter_device_ptr = globalCounter_device.get();

    // Count the number of doublets that we need to produce.
    ::alpaka::exec<Acc>(
            queue, workDiv,
            CountDoubletsKernel{},
            m_seedfinder_config,
            g2_view,
            vecmem::get_data(sp_grid_prefix_sum_buff),
            vecmem::get_data(doublet_counter_buffer),
            (*globalCounter_device).m_nMidBot,
            (*globalCounter_device).m_nMidTop
    );
    ::alpaka::wait(queue);

    // Get the summary values per bin.
    // TODO: Copy to device.
    vecmem::unique_alloc_ptr<device::seeding_global_counter>
        globalCounter_host =
            vecmem::make_unique_alloc<device::seeding_global_counter>(
                (m_mr.host != nullptr) ? *(m_mr.host) : m_mr.main);

    if (globalCounter_host->m_nMidBot == 0 ||
        globalCounter_host->m_nMidTop == 0) {
        return {0, m_mr.main};
    }

    // Set up the doublet counter buffers.
    device::device_doublet_collection_types::buffer doublet_buffer_mb = {
        globalCounter_host->m_nMidBot, m_mr.main};
    m_copy.setup(doublet_buffer_mb);
    device::device_doublet_collection_types::buffer doublet_buffer_mt = {
        globalCounter_host->m_nMidTop, m_mr.main};
    m_copy.setup(doublet_buffer_mt);

    // Calculate the number of threads and thread blocks to run the doublet
    // finding kernel for.
    const unsigned int doublet_counter_buffer_size =
        m_copy.get_size(doublet_counter_buffer);
    blocksPerGrid = (doublet_counter_buffer_size + threadsPerBlock - 1) / threadsPerBlock;
    elementsPerThread = 1u;
    workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "Doublet finder buffer of size" << doublet_counter_buffer_size << std::endl;

    // Find all of the spacepoint doublets.
    ::alpaka::exec<Acc>(
            queue, workDiv,
            FindDoubletsKernel{},
            m_seedfinder_config,
            g2_view,
            vecmem::get_data(doublet_counter_buffer),
            vecmem::get_data(doublet_buffer_mb),
            vecmem::get_data(doublet_buffer_mt)
    );
    ::alpaka::wait(queue);

    // Set up the triplet counter buffers
    device::triplet_counter_spM_collection_types::buffer
        triplet_counter_spM_buffer = {doublet_counter_buffer_size, m_mr.main};
    m_copy.setup(triplet_counter_spM_buffer);
    m_copy.memset(triplet_counter_spM_buffer, 0);
    device::triplet_counter_collection_types::buffer
        triplet_counter_midBot_buffer = {globalCounter_host->m_nMidBot,
                                         m_mr.main,
                                         vecmem::data::buffer_type::resizable};
    m_copy.setup(triplet_counter_midBot_buffer);

    // Calculate the number of threads and thread blocks to run the triplet
    // counting kernel for.
    blocksPerGrid = (globalCounter_host->m_nMidBot + threadsPerBlock - 1) / threadsPerBlock;
    elementsPerThread = 1u;
    workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "Triplet counter buffer of size" << globalCounter_host->m_nMidBot << std::endl;

    // Count the number of triplets that we need to produce.
    ::alpaka::exec<Acc>(
            queue, workDiv,
            CountTripletsKernel{},
            m_seedfinder_config,
            g2_view,
            vecmem::get_data(doublet_counter_buffer),
            vecmem::get_data(doublet_buffer_mb),
            vecmem::get_data(doublet_buffer_mt),
            vecmem::get_data(triplet_counter_spM_buffer),
            vecmem::get_data(triplet_counter_midBot_buffer)
    );
    ::alpaka::wait(queue);

    // Calculate the number of threads and thread blocks to run the triplet
    // count reduction kernel for.
    blocksPerGrid = (doublet_counter_buffer_size + threadsPerBlock - 1) / threadsPerBlock;
    elementsPerThread = 1u;
    workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "Triplet finder buffer of size" << doublet_counter_buffer_size << std::endl;

    // Reduce the triplet counts per spM.
    ::alpaka::exec<Acc>(
            queue, workDiv,
            ReduceTripletCounts{},
            vecmem::get_data(doublet_counter_buffer),
            vecmem::get_data(triplet_counter_spM_buffer),
            (*globalCounter_device).m_nTriplets
    );
    ::alpaka::wait(queue);

    // TODO: Copy globalCounter_host / globalCounter_device again.
    // CUDA_ERROR_CHECK(cudaMemcpyAsync(globalCounter_host.get(),
    //                                  globalCounter_device.get(),
    //                                  sizeof(device::seeding_global_counter),
    //                                  cudaMemcpyDeviceToHost, stream));

    // Find all of the spacepoint triplets.
    // kernels::find_triplets<<<nTripletFindBlocks, nTripletFindThreads>>>(
    //     m_seedfinder_config, m_seedfilter_config, g2_view,
    //     doublet_buffers.middleTop, triplet_counter_buffer,
    //     triplet_counter_prefix_sum_buff, triplet_buffer);

    // // Create prefix sum buffer
    // vecmem::data::vector_buffer triplet_prefix_sum_buff = make_prefix_sum_buff(
    //     m_copy->get_sizes(triplet_buffer.items), *m_copy, m_mr);

    // Calculate the number of threads and thread blocks to run the weight
    // updating kernel for.
    // const unsigned int nWeightUpdatingThreads = WARP_SIZE * 2;
    // const unsigned int nWeightUpdatingBlocks =
    //     (triplet_prefix_sum_buff.size() + nWeightUpdatingThreads - 1) /
    //     nWeightUpdatingThreads;

    // Update the weights of all spacepoint triplets.
    // kernels::update_triplet_weights<<<
    //     nWeightUpdatingBlocks, nWeightUpdatingThreads,
    //     sizeof(scalar) * m_seedfilter_config.compatSeedLimit *
    //         nWeightUpdatingThreads>>>(m_seedfilter_config, g2_view,
    //                                   triplet_prefix_sum_buff, triplet_buffer);

    // // Take header of the triplet counter container buffer into host
    // vecmem::vector<device::triplet_counter_header> tcc_headers(
    //     m_mr.host ? m_mr.host : &(m_mr.main));
    // (*m_copy)(triplet_counter_buffer.headers, tcc_headers);

    // Get the number of seeds (triplets)
    // unsigned int n_triplets = 0;
    // for (const auto& h : tcc_headers) {
    //     n_triplets += h.m_nTriplets;
    // }

    // seed_collection_types::buffer seed_buffer(
    //     globalCounter_host.m_nTriplets, m_mr.main,
    //     vecmem::data::buffer_type::resizable);
    // m_copy.setup(seed_buffer);

    // Calculate the number of threads and thread blocks to run the seed
    // selecting kernel for.
    // const unsigned int nSeedSelectingThreads = WARP_SIZE * 2;
    // const unsigned int nSeedSelectingBlocks =
    //     (doublet_prefix_sum_buff.size() + nSeedSelectingThreads - 1) /
    //     nSeedSelectingThreads;

    // Create seeds out of selected triplets
    // kernels::select_seeds<<<nSeedSelectingBlocks, nSeedSelectingThreads,
    //                         sizeof(triplet) *
    //                             m_seedfilter_config.max_triplets_per_spM *
    //                             nSeedSelectingThreads>>>(
    //     m_seedfilter_config, spacepoints_view, g2_view, doublet_prefix_sum_buff,
    //     doublet_counter_buffer, triplet_buffer, seed_buffer);

    // return seed_buffer;
    return seed_collection_types::buffer();
}

}  // namespace traccc::alpaka
