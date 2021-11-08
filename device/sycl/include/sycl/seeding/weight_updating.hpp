/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <CL/sycl.hpp>

#include "sycl/seeding/detail/triplet_counter.hpp"
#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/seeding_config.hpp>
#include <seeding/detail/triplet.hpp>

namespace traccc {
namespace sycl {

/// Forward declaration of weight updating function
/// The weight of triplets are updated by iterating over triplets which share
/// the same middle spacepoint
///
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param triplet_counter_container vecmem container for triplet counters
/// @param triplet_container vecmem container for triplets
/// @param resource vecmem memory resource
void weight_updating(const seedfilter_config& filter_config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_triplet_counter_container& triplet_counter_container,
                     host_triplet_container& triplet_container,
                     vecmem::memory_resource* resource,
                     ::sycl::queue* q);

// Short aliast for accessor to local memory (shared memory in CUDA)
template <typename T>
using local_accessor = ::sycl::accessor<
    T,
    1,
    ::sycl::access::mode::read_write,
    ::sycl::access::target::local>;

class WeightUpdate {
public:
    WeightUpdate(const seedfilter_config& filter_config,
                internal_spacepoint_container_view internal_sp_view,
                triplet_counter_container_view triplet_counter_view,
                triplet_container_view triplet_view,
                local_accessor<scalar> localMem)
    : m_filter_config(filter_config),
      m_internal_sp_view(internal_sp_view),
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
        for (unsigned int i = 0; i < triplet_device.size(); ++i) {
            nblocks_per_header = triplet_device.get_headers()[i] / groupDim + 1;
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

        auto compat_seedR = m_localMem;
        item.barrier();

        // index of triplet in the item vector
        auto tr_idx = (groupIdx - ref_block_idx) * groupDim + workItemIdx;
        auto& triplet = triplets_per_bin[tr_idx];
        auto& spB_idx = triplet.sp1;
        auto& spM_idx = triplet.sp2;
        auto& spT_idx = triplet.sp3;

        // prevent overflow
        if (tr_idx >= num_triplets_per_bin) {
            return;
        }
        
        // find the reference index (start and end) of the triplet container item
        // vector
        unsigned int start_idx = 0;
        unsigned int end_idx = 0;

        for (auto triplet_counter : triplet_counter_per_bin) {
            end_idx += triplet_counter.n_triplets;

            if (triplet_counter.mid_bot_doublet.sp1 == spM_idx &&
                triplet_counter.mid_bot_doublet.sp2 == spB_idx) {
                break;
            }

            start_idx += triplet_counter.n_triplets;
        }

        if (end_idx >= triplets_per_bin.size()) {
            end_idx = fmin(triplets_per_bin.size(), end_idx);
        }

        // prevent overflow
        if (start_idx >= triplets_per_bin.size()) {
            return;
        }

        auto& current_spT =
        internal_sp_device.get_items()[spT_idx.bin_idx][spT_idx.sp_idx];

        scalar currentTop_r = current_spT.radius();

        // if two compatible seeds with high distance in r are found, compatible
        // seeds span 5 layers
        // -> very good seed
        scalar lowerLimitCurv =
            triplet.curvature - m_filter_config.deltaInvHelixDiameter;
        scalar upperLimitCurv =
            triplet.curvature + m_filter_config.deltaInvHelixDiameter;
        int num_compat_seedR = 0;

        // iterate over triplets
        for (auto tr_it = triplets_per_bin.begin() + start_idx;
            tr_it != triplets_per_bin.begin() + end_idx; tr_it++) {
            if (triplet == *tr_it) {
                continue;
            }
            auto& other_triplet = *tr_it;
            auto other_spT_idx = (*tr_it).sp3;
            auto other_spT =
                internal_sp_device
                    .get_items()[other_spT_idx.bin_idx][other_spT_idx.sp_idx];

            // compared top SP should have at least deltaRMin distance
            scalar otherTop_r = other_spT.radius();
            scalar deltaR = currentTop_r - otherTop_r;
            if (std::abs(deltaR) < m_filter_config.deltaRMin) {
                continue;
            }

            // curvature difference within limits?
            // TODO: how much slower than sorting all vectors by curvature
            // and breaking out of loop? i.e. is vector size large (e.g. in
            // jets?)
            if (other_triplet.curvature < lowerLimitCurv) {
                continue;
            }
            if (other_triplet.curvature > upperLimitCurv) {
                continue;
            }

            bool newCompSeed = true;

            for (unsigned int i_s = 0; i_s < num_compat_seedR; ++i_s) {
                scalar previousDiameter = compat_seedR[i_s];

                // original ATLAS code uses higher min distance for 2nd found
                // compatible seed (20mm instead of 5mm) add new compatible seed
                // only if distance larger than rmin to all other compatible
                // seeds
                if (std::abs(previousDiameter - otherTop_r) <
                    m_filter_config.deltaRMin) {
                    newCompSeed = false;
                    break;
                }
            }

            if (newCompSeed) {
                compat_seedR[num_compat_seedR] = otherTop_r;
                triplet.weight += m_filter_config.compatSeedWeight;
                num_compat_seedR++;
            }

            if (num_compat_seedR >= m_filter_config.compatSeedLimit) {
                break;
            }
        }

      }
private:
    const seedfilter_config m_filter_config;
    internal_spacepoint_container_view m_internal_sp_view;
    triplet_counter_container_view m_triplet_counter_view;
    triplet_container_view m_triplet_view;
    local_accessor<scalar> m_localMem;
};

}  // namespace sycl
}  // namespace traccc
