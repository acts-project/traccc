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
/// Forward declaration of doublet counting function
/// The number of mid-bot and mid-top doublets are counted for all spacepoints
/// and recorded into doublet counter container if the number of doublets are
/// larger than zero.
///
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param resource vecmem memory resource
/// @param q sycl queue for kernel scheduling
void doublet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      vecmem::memory_resource* resource,
                      ::sycl::queue* q);

// Kernel class for doublet counting
class DupletCount {
public:
    DupletCount(const seedfinder_config config,
               internal_spacepoint_container_view internal_sp_view, 
                doublet_counter_container_view doublet_counter_view)
    : m_config(config),
      m_internal_sp_view(internal_sp_view),
      m_doublet_counter_view(doublet_counter_view) {} 

    void operator()(::sycl::nd_item<1> item) const {
        
        // Equivalent to blockIdx.x in cuda
        auto groupIdx = item.get_group(0);
        // Equivalent to blockDim.x in cuda
        auto groupDim = item.get_local_range(0);
        // Equivalent to threadIdx.x in cuda
        auto workItemIdx = item.get_local_id(0);
        
        // Get device container for input parameters
        device_internal_spacepoint_container internal_sp_device(
            {m_internal_sp_view.headers, m_internal_sp_view.items});
        device_doublet_counter_container doublet_counter_device(
            {m_doublet_counter_view.headers, m_doublet_counter_view.items});
        
        // Get the bin index of spacepoint binning and reference block idx for the
        // bin index
        unsigned int bin_idx = 0;
        unsigned int ref_block_idx = 0;

        /////////////// TAken from CUDA helper function ///////////////////////
        // the item jagged vector of edm
        auto jag_vec = internal_sp_device.get_items();

        /// number of blocks accumulated upto current header idx
        unsigned int nblocks_accum = 0;

        /// number of blocks for one header entry
        unsigned int nblocks_per_header = 0;

        // taken from cuda helper functions - get_header_idx()
        for (unsigned int i = 0; i < jag_vec.size(); ++i) {
            nblocks_per_header = jag_vec[i].size() / groupDim + 1;
            nblocks_accum += nblocks_per_header;

            if (groupIdx < nblocks_accum) {
                bin_idx = i;
                break;
            }
            ref_block_idx += nblocks_per_header;
        }
        /////////////////// End of the helper funciton /////////////////////////   

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

    // index of internal spacepoint in the item vector
    auto sp_idx = (groupIdx - ref_block_idx) * groupDim + workItemIdx;

    if (sp_idx >= doublet_counter_per_bin.size()) return;

    // zero initialization for the number of doublets per thread (or middle sp)
    unsigned int n_mid_bot = 0;
    unsigned int n_mid_top = 0;

    // zero initialization for the number of doublets per bin
    doublet_counter_per_bin[sp_idx].n_mid_bot = 0;
    doublet_counter_per_bin[sp_idx].n_mid_top = 0;

    // middle spacepoint index
    auto spM_loc = sp_location({bin_idx, static_cast<uint32_t>(sp_idx)});
    // middle spacepoint
    const auto& isp = internal_sp_per_bin[sp_idx];

    // Loop over (bottom and top) internal spacepoints in the neighbor bins
    for (size_t i_n = 0; i_n < bin_info.bottom_idx.counts; ++i_n) {
        const auto& neigh_bin = bin_info.bottom_idx.vector_indices[i_n];
        const auto& neigh_internal_sp_per_bin =
            internal_sp_device.get_items().at(neigh_bin);

        for (size_t spB_idx = 0; spB_idx < neigh_internal_sp_per_bin.size();
            ++spB_idx) {
            const auto& neigh_isp = neigh_internal_sp_per_bin[spB_idx];

            // Check if middle and bottom sp can form a doublet
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, m_config,
                                                    true)) {
                n_mid_bot++;
            }

            // Check if middle and top sp can form a doublet
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, m_config,
                                                    false)) {
                n_mid_top++;
            }
        }
    }
    // if number of mid-bot and mid-top doublet for a middle spacepoint is
    // larger than 0, the entry is added to the doublet counter
    if (n_mid_bot > 0 && n_mid_top > 0) {
        auto pos = atomic_add(&num_compat_spM_per_bin, 1);
        // auto obj = ::sycl::atomic<uint32_t, ::sycl::access::address_space::global_space>(&num_compat_spM_per_bin);
        // auto pos = obj.fetch_add(1);
        doublet_counter_per_bin[pos] = {spM_loc, n_mid_bot, n_mid_top};
    }        
}
private: 
    const seedfinder_config m_config;
    internal_spacepoint_container_view m_internal_sp_view;
    doublet_counter_container_view m_doublet_counter_view;
};

} // namespace sycl
} // namespace traccc