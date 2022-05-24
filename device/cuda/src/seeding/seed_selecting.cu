/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/seeding/seed_selecting.hpp"
#include "traccc/cuda/utils/cuda_helper.cuh"

// Thrust include(s).
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

// System include(s).
#include <algorithm>

namespace traccc {
namespace cuda {

/// Forward declaration of seed selecting kernel
/// The good triplets are selected and recorded into seed container
///
/// @param filter_config seed filter config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param doublet_counter_container vecmem container for doublet_counter
/// @param triplet_counter_container vecmem container for triplet counters
/// @param triplet_container vecmem container for triplets
/// @param seed_container vecmem container for seeds
__global__ void seed_selecting_kernel(
    const seedfilter_config filter_config,
    spacepoint_container_types::const_view spacepoints_view,
    sp_grid_const_view internal_sp_view,
    device::doublet_counter_container_types::const_view doublet_counter_view,
    triplet_counter_container_view triplet_counter_view,
    triplet_container_view triplet_view,
    vecmem::data::vector_view<seed> seed_view);

void seed_selecting(
    const seedfilter_config& filter_config,
    const vecmem::vector<device::doublet_counter_header>& dcc_headers,
    const spacepoint_container_types::host& spacepoints,
    sp_grid_const_view internal_sp_view,
    device::doublet_counter_container_types::const_view dcc_view,
    triplet_counter_container_view tcc_view, triplet_container_view tc_view,
    vecmem::data::vector_buffer<seed>& seed_buffer,
    vecmem::memory_resource& resource) {

    unsigned int nbins = internal_sp_view._data_view.m_size;

    spacepoint_container_types::const_data spacepoints_view =
        get_data(spacepoints, &resource);

    // The thread-block is desinged to make each thread investigate the
    // compatible middle spacepoint

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 1;

    // -- Num blocks
    // The dimension of grid is = sum_i{N_i}, where:
    // i is the spacepoint bin index
    // N_i is the number of blocks for i-th bin, defined as num_triplets_per_bin
    // / num_threads + 1
    unsigned int num_blocks = 0;
    for (size_t i = 0; i < nbins; ++i) {
        num_blocks += dcc_headers[i].m_nSpM / num_threads + 1;
    }

    // shared memory assignment for the triplets of a compatible middle
    // spacepoint
    unsigned int sh_mem =
        sizeof(triplet) * num_threads * filter_config.max_triplets_per_spM;

    // run the kernel
    seed_selecting_kernel<<<num_blocks, num_threads, sh_mem>>>(
        filter_config, spacepoints_view, internal_sp_view, dcc_view, tcc_view,
        tc_view, seed_buffer);
    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void seed_selecting_kernel(
    const seedfilter_config filter_config,
    spacepoint_container_types::const_view spacepoints_view,
    sp_grid_const_view internal_sp_view,
    device::doublet_counter_container_types::const_view doublet_counter_view,
    triplet_counter_container_view triplet_counter_view,
    triplet_container_view triplet_view,
    vecmem::data::vector_view<seed> seed_view) {

    // Get device container for input parameters
    const spacepoint_container_types::const_device spacepoints_device(
        spacepoints_view);
    const_sp_grid_device internal_sp_device(internal_sp_view);

    const device::doublet_counter_container_types::const_device
        doublet_counter_device(doublet_counter_view);
    device_triplet_counter_container triplet_counter_device(
        triplet_counter_view);
    device_triplet_container triplet_device(triplet_view);
    device_seed_collection seed_device(seed_view);

    // Get the bin and item index
    unsigned int bin_idx(0), item_idx(0);
    cuda_helper::find_idx_on_container(doublet_counter_device, bin_idx,
                                       item_idx);
    // Header of internal spacepoint container : spacepoint bin information
    // Item of internal spacepoint container : internal spacepoint objects per
    // bin
    auto internal_sp_per_bin = internal_sp_device.bin(bin_idx);
    auto& num_compat_spM_per_bin =
        doublet_counter_device.get_headers().at(bin_idx).m_nSpM;

    // Header of doublet counter : number of compatible middle sp per bin
    // Item of doublet counter : doublet counter objects per bin
    auto doublet_counter_per_bin =
        doublet_counter_device.get_items().at(bin_idx);

    // Header of triplet counter: number of compatible mid_top doublets per bin
    // Item of triplet counter: triplet counter objects per bin
    auto& num_compat_mb_per_bin =
        triplet_counter_device.get_headers().at(bin_idx).n_mid_bot;

    // Header of triplet: number of triplets per bin
    // Item of triplet: triplet objects per bin
    auto& num_triplets_per_bin =
        triplet_device.get_headers().at(bin_idx).n_triplets;
    auto triplets_per_bin = triplet_device.get_items().at(bin_idx);

    extern __shared__ triplet triplets_per_spM[];

    // prevent overflow
    if (item_idx >= num_compat_spM_per_bin) {
        return;
    }

    // middle spacepoint index
    auto& spM_loc = doublet_counter_per_bin[item_idx].m_spM;
    auto& spM_idx = spM_loc.sp_idx;
    // middle spacepoint
    auto& spM = internal_sp_per_bin[spM_idx];

    // number of triplets per compatible middle spacepoint
    unsigned int n_triplets_per_spM = 0;

    // the start index of triplets_per_spM
    unsigned int stride = threadIdx.x * filter_config.max_triplets_per_spM;

    // iterate over the triplets in the bin
    for (unsigned int i = 0; i < num_triplets_per_bin; ++i) {
        auto& aTriplet = triplets_per_bin[i];
        auto& spB_loc = aTriplet.sp1;
        auto& spT_loc = aTriplet.sp3;
        auto& spB = internal_sp_device.bin(spB_loc.bin_idx)[spB_loc.sp_idx];
        auto& spT = internal_sp_device.bin(spT_loc.bin_idx)[spT_loc.sp_idx];

        // consider only the triplets with the same middle spacepoint
        if (spM_loc == aTriplet.sp2) {

            // update weight of triplet
            seed_selecting_helper::seed_weight(filter_config, spM, spB, spT,
                                               aTriplet.weight);

            // check if it is a good triplet
            if (!seed_selecting_helper::single_seed_cut(filter_config, spM, spB,
                                                        spT, aTriplet.weight)) {
                continue;
            }

            // if the number of good triplets is larger than the threshold, the
            // triplet with the lowest weight is removed
            if (n_triplets_per_spM >= filter_config.max_triplets_per_spM) {
                int begin_idx = stride;
                int end_idx = stride + filter_config.max_triplets_per_spM;

                // Note: min_index method gives a result different
                //       from sorting method when there are the cases where
                //       weight & z_vertex are same.
                //
                //       So min_index method reduces seed matching ratio
                //       since the cpu version is using sorting method.
                //
                //       But that doesn't mean min_index method
                //       is wrong of course
                //
                //       Let's not be so obsessed about achieving
                //       perfectly same result :))))))))

                int min_index =
                    std::min_element(triplets_per_spM + begin_idx,
                                     triplets_per_spM + end_idx,
                                     [&](triplet& lhs, triplet& rhs) {
                                         return lhs.weight < rhs.weight;
                                     }) -
                    triplets_per_spM;

                auto& min_weight = triplets_per_spM[min_index].weight;

                if (aTriplet.weight > min_weight) {
                    triplets_per_spM[min_index] = aTriplet;
                }

                // (deprecated) sorting method -> good for seed matching ratio
                // but slow
                /*
                  thrust::sort(thrust::seq,
                  triplets_per_spM+begin_idx,
                  triplets_per_spM+end_idx,
                  triplet_weight_descending());

                  if (aTriplet.weight >= triplets_per_spM[end_idx-1].weight){
                  triplets_per_spM[end_idx-1] = aTriplet;
                  }
                */
            }

            else if (n_triplets_per_spM < filter_config.max_triplets_per_spM) {
                triplets_per_spM[stride + n_triplets_per_spM] = aTriplet;
                n_triplets_per_spM++;
            }
        }
    }

    // sort the triplets per spM
    // sequential version of thrust sorting algorithm is used
    thrust::sort(
        thrust::seq, triplets_per_spM + stride,
        triplets_per_spM + stride + n_triplets_per_spM,
        [&](triplet& lhs, triplet& rhs) {
            if (lhs.weight != rhs.weight) {
                return lhs.weight > rhs.weight;
            } else {

                scalar seed1_sum = 0;
                scalar seed2_sum = 0;

                auto& ispB1 =
                    internal_sp_device.bin(lhs.sp1.bin_idx)[lhs.sp1.sp_idx];
                auto& ispT1 =
                    internal_sp_device.bin(lhs.sp3.bin_idx)[lhs.sp3.sp_idx];
                auto& ispB2 =
                    internal_sp_device.bin(rhs.sp1.bin_idx)[rhs.sp1.sp_idx];
                auto& ispT2 =
                    internal_sp_device.bin(rhs.sp3.bin_idx)[rhs.sp3.sp_idx];

                const auto& spB1 = spacepoints_device.at(ispB1.m_link);
                const auto& spT1 = spacepoints_device.at(ispT1.m_link);
                const auto& spB2 = spacepoints_device.at(ispB2.m_link);
                const auto& spT2 = spacepoints_device.at(ispT2.m_link);

                seed1_sum += pow(spB1.y(), 2) + pow(spB1.z(), 2);
                seed1_sum += pow(spT1.y(), 2) + pow(spT1.z(), 2);
                seed2_sum += pow(spB2.y(), 2) + pow(spB2.z(), 2);
                seed2_sum += pow(spT2.y(), 2) + pow(spT2.z(), 2);

                return seed1_sum > seed2_sum;
            }
        });
    // triplet_weight_descending());

    // the number of good seed per compatible middle spacepoint
    unsigned int n_seeds_per_spM = 0;

    // iterate over the good triplets for final selection of seeds
    for (unsigned int i = stride; i < stride + n_triplets_per_spM; ++i) {
        auto& aTriplet = triplets_per_spM[i];
        auto& spB_loc = aTriplet.sp1;
        auto& spT_loc = aTriplet.sp3;
        auto& spB = internal_sp_device.bin(spB_loc.bin_idx)[spB_loc.sp_idx];
        auto& spT = internal_sp_device.bin(spT_loc.bin_idx)[spT_loc.sp_idx];

        // if the number of seeds reaches the threshold, break
        if (n_seeds_per_spM >= filter_config.maxSeedsPerSpM + 1) {
            break;
        }

        seed aSeed{spB.m_link, spM.m_link, spT.m_link, aTriplet.weight,
                   aTriplet.z_vertex};

        // check if it is a good triplet
        if (seed_selecting_helper::cut_per_middle_sp(
                filter_config, spacepoints_device, aSeed, aTriplet.weight) ||
            n_seeds_per_spM == 0) {

            n_seeds_per_spM++;

            seed_device.push_back(aSeed);
        }
    }
}

}  // namespace cuda
}  // namespace traccc
