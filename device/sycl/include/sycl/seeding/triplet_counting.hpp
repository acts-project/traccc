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
#include "sycl/seeding/detail/sycl_helper.hpp"
#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/doublet.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/triplet.hpp>
#include <seeding/doublet_finding_helper.hpp>
#include <seeding/triplet_finding_helper.hpp>


namespace traccc {
namespace sycl {

/// Forward declaration of triplet counting function
/// The number of triplets per mid-bot doublets are counted and recorded into
/// triplet counter container
///
/// @param config seed finder config
/// @param internal_sp_view vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param mid_bot_doublet_container vecmem container for mid-bot doublets
/// @param mid_top_doublet_container vecmem container for mid-top doublets
/// @param triplet_counter_container vecmem container for triplet counters
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void triplet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      host_doublet_container& mid_bot_doublet_container,
                      host_doublet_container& mid_top_doublet_container,
                      host_triplet_counter_container& triplet_counter_container,
                      vecmem::memory_resource* resource,
                      ::sycl::queue* q);
    
// Kernel class for triplet counting
class TripletCount {
public:
    TripletCount(const seedfinder_config& config,
                internal_spacepoint_container_view internal_sp_view,
                doublet_counter_container_view doublet_counter_view,
                doublet_container_view mid_bot_doublet_view,
                doublet_container_view mid_top_doublet_view,
                triplet_counter_container_view triplet_counter_view) 
    : m_config(config),
      m_internal_sp_view(internal_sp_view),
      m_doublet_counter_view(doublet_counter_view),
      m_mid_bot_doublet_view(mid_bot_doublet_view),
      m_mid_top_doublet_view(mid_top_doublet_view),
      m_triplet_counter_view(triplet_counter_view) {}
    
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
    
        // Get the bin index of spacepoint binning and reference block idx for the
        // bin index
        unsigned int bin_idx = 0;
        unsigned int ref_block_idx = 0;

       /////////////// TAken from CUDA helper function ///////////////////////
       /// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < mid_bot_doublet_device.size(); ++i) {
            nblocks_per_header = mid_bot_doublet_device.get_headers()[i] / groupDim + 1;
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
    
        // index of middle-bot doublet in the item vector
        auto mb_idx = (groupIdx - ref_block_idx) * groupDim + workItemIdx;

        // prevent the tail threads referring the null doublet counter
        if (mb_idx >= num_mid_bot_doublets_per_bin) return;

        // middle-bot doublet
        const auto& mid_bot_doublet = mid_bot_doublets_per_bin[mb_idx];
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
        if (mt_start_idx >= mid_top_doublets_per_bin.size()) return;
        // number of triplets per thread (or per middle-bot doublet)
        unsigned int num_triplets_per_mb = 0;

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
                num_triplets_per_mb++;
            }
        }
        // if the number of triplets per mb is larger than 0, write the triplet
        // counter into the container
        if (num_triplets_per_mb > 0) {
            auto pos = atomic_add(&num_compat_mb_per_bin, 1);
            triplet_counter_per_bin[pos] = {mid_bot_doublet, num_triplets_per_mb};
        }
                
    }
private:
    const seedfinder_config m_config;
    internal_spacepoint_container_view m_internal_sp_view;
    doublet_counter_container_view m_doublet_counter_view;
    doublet_container_view m_mid_bot_doublet_view;
    doublet_container_view m_mid_top_doublet_view;
    triplet_counter_container_view m_triplet_counter_view;
};

} // namespace sycl
} // namespace traccc