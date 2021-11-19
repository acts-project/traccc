/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include <CL/sycl.hpp>

#include "sycl/seeding/detail/doublet_counter.hpp"
#include "sycl/seeding/detail/sycl_helper.hpp"
#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/doublet_finding_helper.hpp>

#include "seeding/detail/doublet.hpp"


namespace traccc {
namespace sycl {
/// Forward declaration of doublet finding function
/// The mid-bot and mid-top doublets are found for the compatible middle
/// spacepoints which was recorded by doublet_counting
///
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param mid_bot_doublet_container vecmem container for mid-bot doublets
/// @param mid_top_doublet_container vecmem container for mid-top doublets
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void doublet_finding(const seedfinder_config& config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_doublet_counter_container& doublet_counter_container,
                     host_doublet_container& mid_bot_doublet_container,
                     host_doublet_container& mid_top_doublet_container,
                     vecmem::memory_resource* resource,
                     ::sycl::queue* q);

// Short aliast for accessor to local memory (shared memory in CUDA)
template <typename T>
using local_accessor = ::sycl::accessor<
    T,
    1,
    ::sycl::access::mode::read_write,
    ::sycl::access::target::local>;

// kernel class for doublet finding
class DupletFind {
public:
    DupletFind(const seedfinder_config& config,
                internal_spacepoint_container_view internal_sp_view,
                doublet_counter_container_view doublet_counter_view,
                doublet_container_view mid_bot_doublet_view,
                doublet_container_view mid_top_doublet_view,
                local_accessor<int> localMem)
    : m_config(config),
      m_internal_sp_view(internal_sp_view),
      m_doublet_counter_view(doublet_counter_view),
      m_mid_bot_doublet_view(mid_bot_doublet_view),
      m_mid_top_doublet_view(mid_top_doublet_view),
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
        
        // Get the bin index of spacepoint binning and reference block idx for the
        // bin index
        unsigned int bin_idx = 0;
        unsigned int ref_block_idx = 0;

       /////////////// TAken from CUDA helper function ///////////////////////
       /// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;
        for (unsigned int i = 0; i < doublet_counter_device.size(); ++i) {
            nblocks_per_header = doublet_counter_device.get_headers()[i] / groupDim + 1;
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
        const auto& bin_info = internal_sp_device.get_headers().at(bin_idx);
        auto internal_sp_per_bin = internal_sp_device.get_items().at(bin_idx);

        // Header of doublet counter : number of compatible middle sp per bin
        // Item of doublet counter : doublet counter objects per bin
        auto& num_compat_spM_per_bin =
            doublet_counter_device.get_headers().at(bin_idx);
        auto doublet_counter_per_bin =
            doublet_counter_device.get_items().at(bin_idx);

        // Header of doublet: number of mid_bot doublets per bin
        // Item of doublet: doublet objects per bin
        auto& num_mid_bot_doublets_per_bin =
            mid_bot_doublet_device.get_headers().at(bin_idx);
        auto mid_bot_doublets_per_bin =
            mid_bot_doublet_device.get_items().at(bin_idx);

        // Header of doublet: number of mid_top doublets per bin
        // Item of doublet: doublet objects per bin
        auto& num_mid_top_doublets_per_bin =
            mid_top_doublet_device.get_headers().at(bin_idx);
        auto mid_top_doublets_per_bin =
            mid_top_doublet_device.get_items().at(bin_idx);        

        auto num_mid_bot_doublets_per_thread = m_localMem;
        auto num_mid_top_doublets_per_thread = &num_mid_bot_doublets_per_thread[groupDim];
        num_mid_bot_doublets_per_thread[workItemIdx] = 0;
        num_mid_top_doublets_per_thread[workItemIdx] = 0;

        // Convenient alias for the number of doublets per thread
        auto& n_mid_bot_per_spM = num_mid_bot_doublets_per_thread[workItemIdx];
        auto& n_mid_top_per_spM = num_mid_top_doublets_per_thread[workItemIdx];

        // index of doublet counter in the item vector
        auto gid = (groupIdx - ref_block_idx) * groupDim + workItemIdx;
    
        // prevent the tail threads referring the null doublet counter
        if (gid >= num_compat_spM_per_bin) return;

        // index of internal spacepoint in the item vector
        auto sp_idx = doublet_counter_per_bin[gid].spM.sp_idx;
        // middle spacepoint index
        auto spM_loc = sp_location({bin_idx, sp_idx});
        // middle spacepoint
        auto& isp = internal_sp_per_bin[sp_idx];

        // find the reference (start) index of the doublet container item vector,
        // where the doublets are recorded The start index is calculated by
        // accumulating the number of doublets of all previous compatible middle
        // spacepoints
        unsigned int mid_bot_start_idx = 0;
        unsigned int mid_top_start_idx = 0;
        for (unsigned int i = 0; i < gid; i++) {
            mid_bot_start_idx += doublet_counter_per_bin[i].n_mid_bot;
            mid_top_start_idx += doublet_counter_per_bin[i].n_mid_top;
        }
        // Loop over (bottom and top) internal spacepoints in tje neighbor bins
        for (unsigned int i_n = 0; i_n < bin_info.bottom_idx.counts; ++i_n) {
            const auto& neigh_bin = bin_info.bottom_idx.vector_indices[i_n];
            const auto& neigh_internal_sp_per_bin =
                internal_sp_device.get_items().at(neigh_bin);

            for (unsigned int spB_idx = 0;
                spB_idx < neigh_internal_sp_per_bin.size(); ++spB_idx) {
                const auto& neigh_isp = neigh_internal_sp_per_bin[spB_idx];

                // Check if middle and bottom sp can form a doublet
                if (doublet_finding_helper::isCompatible(isp, neigh_isp, m_config,
                                                        true)) {
                    auto spB_loc = sp_location({neigh_bin, spB_idx});

                    // Check conditions
                    // 1) number of mid-bot doublets per spM should be smaller than
                    // what is counted in doublet_counting (so it should be true
                    // always) 2) prevent overflow
                    if (n_mid_bot_per_spM <
                            doublet_counter_per_bin[gid].n_mid_bot &&
                        num_mid_bot_doublets_per_bin <
                            mid_bot_doublets_per_bin.size()) {
                        unsigned int pos = mid_bot_start_idx + n_mid_bot_per_spM;

                        // prevent overflow again
                        if (pos >= mid_bot_doublets_per_bin.size()) {
                            continue;
                        }

                        // write the doublet into the container
                        mid_bot_doublets_per_bin[pos] = doublet({spM_loc, spB_loc});
                        n_mid_bot_per_spM++;
                    }
                }

                // Check if middle and top sp can form a doublet
                if (doublet_finding_helper::isCompatible(isp, neigh_isp, m_config,
                                                        false)) {
                    auto spT_loc = sp_location({neigh_bin, spB_idx});

                    // Check conditions
                    // 1) number of mid-top doublets per spM should be smaller than
                    // what is counted in doublet_counting (so it should be true
                    // always) 2) prevent overflow
                    if (n_mid_top_per_spM <
                            doublet_counter_per_bin[gid].n_mid_top &&
                        num_mid_top_doublets_per_bin <
                            mid_top_doublets_per_bin.size()) {
                        unsigned int pos = mid_top_start_idx + n_mid_top_per_spM;

                        // prevent overflow again
                        if (pos >= mid_top_doublets_per_bin.size()) {
                            continue;
                        }

                        // write the doublet into the container
                        mid_top_doublets_per_bin[pos] = doublet({spM_loc, spT_loc});
                        n_mid_top_per_spM++;
                    }
                }
            }
        }
        // Calculate the number doublets per "block" with reducing sum technique
        item.barrier();
        auto bottom_result = ::sycl::reduce_over_group(workGroup, num_mid_bot_doublets_per_thread[workItemIdx], ::sycl::ext::oneapi::plus<>());
        auto top_result = ::sycl::reduce_over_group(workGroup, num_mid_top_doublets_per_thread[workItemIdx], ::sycl::ext::oneapi::plus<>());

        // Calculate the number doublets per bin by atomic-adding the number of
        // doublets per block
        if (workItemIdx == 0) {
            atomic_add(&num_mid_bot_doublets_per_bin, bottom_result);
            atomic_add(&num_mid_top_doublets_per_bin, top_result);
        }
    }
private:
    const seedfinder_config m_config;
    internal_spacepoint_container_view m_internal_sp_view;
    doublet_counter_container_view m_doublet_counter_view;
    doublet_container_view m_mid_bot_doublet_view;
    doublet_container_view m_mid_top_doublet_view;
    local_accessor<int> m_localMem;
};

} // namespace sycl
} // namespace traccc