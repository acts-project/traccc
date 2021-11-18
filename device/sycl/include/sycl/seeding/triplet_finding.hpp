/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include <CL/sycl.hpp>

#include "sycl/seeding/detail/doublet_counter.hpp"
#include "sycl/seeding/detail/triplet_counter.hpp"
#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/doublet.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/doublet_finding_helper.hpp>
#include <seeding/seed_selecting_helper.hpp>
#include <seeding/triplet_finding_helper.hpp>

namespace traccc {
namespace sycl {

/// Forward declaration of triplet finding function
/// The triplets per mid-bot doublets are found for the compatible mid-bot
/// doublets which were recorded during triplet_counting
///
/// @param config seed finder config
/// @param filter_config seed filter config
/// @param internal_sp_view vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param mid_bot_doublet_container vecmem container for mid-bot doublets
/// @param mid_top_doublet_container vecmem container for mid-top doublets
/// @param triplet_counter_container vecmem container for triplet counters
/// @param triplet_container vecmem container for triplets
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void triplet_finding(const seedfinder_config& config,
                     const seedfilter_config& filter_config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_doublet_counter_container& doublet_counter_container,
                     host_doublet_container& mid_bot_doublet_container,
                     host_doublet_container& mid_top_doublet_container,
                     host_triplet_counter_container& triplet_counter_container,
                     host_triplet_container& triplet_container,
                     vecmem::memory_resource* resource,
                     ::sycl::queue* q);

// Define shorthand alias for the type of atomics needed by this kernel 
template <typename T>
using global_atomic_ref = ::sycl::ext::oneapi::atomic_ref<
    T,
    ::sycl::ext::oneapi::memory_order::relaxed,
    ::sycl::ext::oneapi::memory_scope::system,
    ::sycl::access::address_space::global_space>;

// Short aliast for accessor to local memory (shared memory in CUDA)
template <typename T>
using local_accessor = ::sycl::accessor<
    T,
    1,
    ::sycl::access::mode::read_write,
    ::sycl::access::target::local>;

class TripletFind {
public:
    TripletFind(const seedfinder_config& config,
                const seedfilter_config& filter_config,
                internal_spacepoint_container_view internal_sp_view,
                doublet_counter_container_view doublet_counter_view,
                doublet_container_view mid_bot_doublet_view,
                doublet_container_view mid_top_doublet_view,
                triplet_counter_container_view triplet_counter_view,
                triplet_container_view triplet_view,
                local_accessor<int> localMem)
    : m_config(config),
      m_filter_config(filter_config),
      m_internal_sp_view(internal_sp_view),
      m_doublet_counter_view(doublet_counter_view),
      m_mid_bot_doublet_view(mid_bot_doublet_view),
      m_mid_top_doublet_view(mid_top_doublet_view),
      m_triplet_counter_view(triplet_counter_view),
      m_triplet_view(triplet_view),
      m_localMem(localMem) {}

    void operator()(::sycl::nd_item<1> item) const {

         // Mapping cuda indexing to dpc++
        auto workGroup = item.get_group();
        
        // Equivalent to blockIdx.x in cuda
        auto groupIdx = workGroup.get_linear_id();
        // Equivalent to blockDim.x in cuda
        auto groupDim = workGroup.get_local_range(0);
        // Equivalent to threadIdx.x in cuda
        auto workItemIdx = item.get_local_linear_id();
    
        device_internal_spacepoint_container internal_sp_device(
        {m_internal_sp_view.headers, m_internal_sp_view.items});
        device_doublet_counter_container doublet_counter_device(
            {m_doublet_counter_view.headers, m_doublet_counter_view.items});
        device_doublet_container mid_bot_doublet_device(
            {m_mid_bot_doublet_view.headers, m_mid_bot_doublet_view.items});
        device_doublet_container mid_top_doublet_device(
            {m_mid_top_doublet_view.headers, m_mid_top_doublet_view.items});

        device_triplet_counter_container triplet_counter_device(
            {m_triplet_counter_view.headers, m_triplet_counter_view.items});
        device_triplet_container triplet_device(
            {m_triplet_view.headers, m_triplet_view.items});
        
        // Get the bin index of spacepoint binning and reference block idx for the
        // bin index
        unsigned int bin_idx = 0;
        unsigned int ref_block_idx = 0;

        /////////////// TAken from CUDA helper function ///////////////////////
        /// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < triplet_counter_device.size(); ++i) {
            nblocks_per_header = triplet_counter_device.get_headers()[i] / groupDim + 1;
            nblocks_accum += nblocks_per_header;

            if (groupIdx < nblocks_accum) {
                bin_idx = i;
                break;
            }
            ref_block_idx += nblocks_per_header;
        }
        /////////////////// End of the helper funciton ////////////////////

        // Header of internal spacepoint container : spacepoint bin information
        // Item of internal spacepoint container : internal spacepoint objects per
        // bin
        auto internal_sp_per_bin = internal_sp_device.get_items().at(bin_idx);
        auto& num_compat_spM_per_bin =
            doublet_counter_device.get_headers().at(bin_idx);
        
        // Header of doublet counter : number of compatible middle sp per bin
        // Item of doublet counter : doublet counter objects per bin
        auto doublet_counter_per_bin =
            doublet_counter_device.get_items().at(bin_idx);

        // Header of doublet: number of mid_bot doublets per bin
        // Item of doublet: doublet objects per bin
        const auto& num_mid_bot_doublets_per_bin =
            mid_bot_doublet_device.get_headers().at(bin_idx);
        auto mid_bot_doublets_per_bin =
            mid_bot_doublet_device.get_items().at(bin_idx);

        // Header of doublet: number of mid_top doublets per bin
        // Item of doublet: doublet objects per bin
        const auto& num_mid_top_doublets_per_bin =
            mid_top_doublet_device.get_headers().at(bin_idx);
        auto mid_top_doublets_per_bin =
            mid_top_doublet_device.get_items().at(bin_idx);

        // Header of triplet counter: number of compatible mid_top doublets per bin
        // Item of triplet counter: triplet counter objects per bin
        auto& num_compat_mb_per_bin =
            triplet_counter_device.get_headers().at(bin_idx);
        auto triplet_counter_per_bin =
            triplet_counter_device.get_items().at(bin_idx);

        // Header of triplet: number of triplets per bin
        // Item of triplet: triplet objects per bin
        auto& num_triplets_per_bin = triplet_device.get_headers().at(bin_idx);
        auto triplets_per_bin = triplet_device.get_items().at(bin_idx);

        auto num_triplets_per_thread = m_localMem;
        num_triplets_per_thread[workItemIdx] = 0;

        // index of triplet counter in the item vector
        auto gid = (groupIdx - ref_block_idx) * groupDim + workItemIdx;
         // prevent the tail threads referring the null triplet counter
        if (gid >= num_compat_mb_per_bin) {
            return;
        }

        // middle-bot doublet
        const auto& mid_bot_doublet = triplet_counter_per_bin[gid].mid_bot_doublet;
        // middle spacepoint index
        const auto& spM_idx = mid_bot_doublet.sp1.sp_idx;
        // middle spacepoint
        const auto& spM = internal_sp_per_bin[spM_idx];
        // bin index of bottom spacepoint
        const auto& spB_bin = mid_bot_doublet.sp2.bin_idx;
        // bottom spacepoint index
        const auto& spB_idx = mid_bot_doublet.sp2.sp_idx;
        // bottom spacepoint
        const auto& spB = internal_sp_device.get_items().at(spB_bin)[spB_idx];

        // Apply the conformal transformation to middle-bot doublet
        auto lb = doublet_finding_helper::transform_coordinates(spM, spB, true);

        // Calculate some physical quantities required for triplet compatibility
        // check
        scalar iSinTheta2 = 1 + lb.cotTheta() * lb.cotTheta();
        scalar scatteringInRegion2 = m_config.maxScatteringAngle2 * iSinTheta2;
        scatteringInRegion2 *= m_config.sigmaScattering * m_config.sigmaScattering;
        scalar curvature, impact_parameter;

        // find the reference (start) index of the mid-top doublet container item
        // vector, where the doublets are recorded The start index is calculated by
        // accumulating the number of mid-top doublets of all previous compatible
        // middle spacepoints
        unsigned int mb_end_idx = 0;
        unsigned int mt_start_idx = 0;
        unsigned int mt_end_idx = 0;
        unsigned int mb_idx;

        // First, find the index of middle-bottom doublet
        for (unsigned int i = 0; i < num_mid_bot_doublets_per_bin; i++) {
            if (mid_bot_doublet == mid_bot_doublets_per_bin[i]) {
                mb_idx = i;
                break;
            }
        }

        for (unsigned int i = 0; i < num_compat_spM_per_bin; ++i) {
            mb_end_idx += doublet_counter_per_bin[i].n_mid_bot;
            mt_end_idx += doublet_counter_per_bin[i].n_mid_top;

            if (mb_end_idx > mb_idx) {
                break;
            }

            mt_start_idx += doublet_counter_per_bin[i].n_mid_top;
        }

        if (mt_end_idx >= mid_top_doublets_per_bin.size()) {
            mt_end_idx = fmin(mid_top_doublets_per_bin.size(), mt_end_idx);
        }

        if (mt_start_idx >= mid_top_doublets_per_bin.size()) {
            return;
        }

        // number of triplets per thread (or per middle-bot doublet)
        unsigned int n_triplets_per_mb = 0;

        // find the reference (start) index of the triplet container item vector,
        // where the triplets are recorded
        unsigned int triplet_start_idx = 0;

        // The start index is calculated by accumulating the number of triplets of
        // all previous compatible middle-bottom doublets
        for (unsigned int i = 0; i < gid; i++) {
            triplet_start_idx += triplet_counter_per_bin[i].n_triplets;
        }

        // iterate over mid-top doublets
        for (unsigned int i = mt_start_idx; i < mt_end_idx; ++i) {
            const auto& mid_top_doublet = mid_top_doublets_per_bin[i];

            const auto& spT_bin = mid_top_doublet.sp2.bin_idx;
            const auto& spT_idx = mid_top_doublet.sp2.sp_idx;
            const auto& spT = internal_sp_device.get_items().at(spT_bin)[spT_idx];
            // Apply the conformal transformation to middle-top doublet
            auto lt =
                doublet_finding_helper::transform_coordinates(spM, spT, false);

            // Check if mid-bot and mid-top doublets can form a triplet
            if (triplet_finding_helper::isCompatible(
                    spM, lb, lt, m_config, iSinTheta2, scatteringInRegion2, curvature,
                    impact_parameter)) {
                unsigned int pos = triplet_start_idx + n_triplets_per_mb;
                // prevent the overflow
                if (pos >= triplets_per_bin.size()) {
                    continue;
                }

                triplets_per_bin[pos] =
                    triplet({mid_bot_doublet.sp2, mid_bot_doublet.sp1,
                            mid_top_doublet.sp2, curvature,
                            -impact_parameter * m_filter_config.impactWeightFactor,
                            lb.Zo()});

                num_triplets_per_thread[workItemIdx]++;
                n_triplets_per_mb++;
            }
        }
        
        // Calculate the number of triplets per "block" with reducing sum technique
        item.barrier();
        auto triplets_result = ::sycl::reduce_over_group(workGroup, num_triplets_per_thread[workItemIdx], ::sycl::ext::oneapi::plus<>());

        // Calculate the number of triplets per bin by atomic-adding the number of
        // triplets per block
        if (workItemIdx == 0) {
            global_atomic_ref<uint32_t>(num_triplets_per_bin) += triplets_result;
        }
    }
private:
    const seedfinder_config m_config;
    const seedfilter_config m_filter_config;
    internal_spacepoint_container_view m_internal_sp_view;
    doublet_counter_container_view m_doublet_counter_view;
    doublet_container_view m_mid_bot_doublet_view;
    doublet_container_view m_mid_top_doublet_view;
    triplet_counter_container_view m_triplet_counter_view;
    triplet_container_view m_triplet_view;
    local_accessor<int> m_localMem;
};

} // namespace traccc
} // namespace sycl