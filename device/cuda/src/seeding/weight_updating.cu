/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/seeding/weight_updating.cuh>
#include <cuda/utils/cuda_helper.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

/// Forward declaration of weight updating kernel
/// The weight of triplets are updated by iterating over triplets which share the same middle spacepoint
///    
/// @param config seed finder config
/// @param internal_sp_container vecmem container for internal spacepoint
/// @param triplet_counter_container vecmem container for triplet counters
/// @param triplet_container vecmem container for triplets    
__global__ void weight_updating_kernel(
    const seedfilter_config filter_config,
    internal_spacepoint_container_view internal_sp_view,
    triplet_counter_container_view triplet_counter_view,
    triplet_container_view triplet_view);

void weight_updating(const seedfilter_config& filter_config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_triplet_counter_container& triplet_counter_container,
                     host_triplet_container& triplet_container,
                     vecmem::memory_resource* resource) {
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto triplet_counter_view = get_data(triplet_counter_container, resource);
    auto triplet_view = get_data(triplet_container, resource);

    // The thread-block is desinged to make each thread update the weight of eac triplet
    
    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)    
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks 
    // The dimension of grid is = sum_i{N_i}, where:
    // i is the spacepoint bin index
    // N_i is the number of blocks for i-th bin, defined as num_triplets_per_bin / num_threads + 1      
    unsigned int num_blocks = 0;
    for (size_t i = 0; i < internal_sp_view.headers.size(); ++i) {
        num_blocks += triplet_container.headers[i] / num_threads + 1;
    }

    // shared memory assignment for the radius of the compatible top spacepoints
    unsigned int sh_mem = sizeof(float) * filter_config.compatSeedLimit;

    // run the kernel    
    weight_updating_kernel<<<num_blocks, num_threads, sh_mem>>>(
        filter_config, internal_sp_view, triplet_counter_view, triplet_view);

    // cuda error check    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void weight_updating_kernel(
    const seedfilter_config filter_config,
    internal_spacepoint_container_view internal_sp_view,
    triplet_counter_container_view triplet_counter_view,
    triplet_container_view triplet_view) {
    device_internal_spacepoint_container internal_sp_device(
        {internal_sp_view.headers, internal_sp_view.items});

    device_triplet_counter_container triplet_counter_device(
        {triplet_counter_view.headers, triplet_counter_view.items});
    device_triplet_container triplet_device(
        {triplet_view.headers, triplet_view.items});

    // Get the bin index of spacepoint binning and reference block idx for the bin index           
    unsigned int bin_idx = 0;
    unsigned int ref_block_idx = 0;
    cuda_helper::get_header_idx(triplet_device, bin_idx, ref_block_idx);

    // Header of internal spacepoint container : spacepoint bin information
    // Item of internal spacepoint container : internal spacepoint objects per bin  
    auto internal_sp_per_bin = internal_sp_device.items.at(bin_idx);

    // Header of triplet counter: number of compatible mid_top doublets per bin
    // Item of triplet counter: triplet counter objects per bin        
    auto& num_compat_mb_per_bin = triplet_counter_device.headers.at(bin_idx);
    auto triplet_counter_per_bin = triplet_counter_device.items.at(bin_idx);

    // Header of triplet: number of triplets per bin
    // Item of triplet: triplet objects per bin            
    auto& num_triplets_per_bin = triplet_device.headers.at(bin_idx);
    auto triplets_per_bin = triplet_device.items.at(bin_idx);

    extern __shared__ float compat_seedR[];
    __syncthreads();

    // index of triplet in the item vector            
    auto tr_idx = (blockIdx.x - ref_block_idx) * blockDim.x + threadIdx.x;
    auto& triplet = triplets_per_bin[tr_idx];
    auto& spB_idx = triplet.sp1;
    auto& spM_idx = triplet.sp2;
    auto& spT_idx = triplet.sp3;

    // prevent overflow
    if (tr_idx >= num_triplets_per_bin) {
        return;
    }

    // find the reference index (start and end) of the triplet container item vector
    size_t start_idx = 0;
    size_t end_idx = 0;

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
        internal_sp_device.items[spT_idx.bin_idx][spT_idx.sp_idx];

    float currentTop_r = current_spT.radius();

    // if two compatible seeds with high distance in r are found, compatible
    // seeds span 5 layers
    // -> very good seed
    float lowerLimitCurv =
        triplet.curvature - filter_config.deltaInvHelixDiameter;
    float upperLimitCurv =
        triplet.curvature + filter_config.deltaInvHelixDiameter;
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
                .items[other_spT_idx.bin_idx][other_spT_idx.sp_idx];

        // compared top SP should have at least deltaRMin distance
        float otherTop_r = other_spT.radius();
        float deltaR = currentTop_r - otherTop_r;
        if (std::abs(deltaR) < filter_config.deltaRMin) {
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

        for (size_t i_s = 0; i_s < num_compat_seedR; ++i_s) {
            float previousDiameter = compat_seedR[i_s];

            // original ATLAS code uses higher min distance for 2nd found
            // compatible seed (20mm instead of 5mm) add new compatible seed
            // only if distance larger than rmin to all other compatible
            // seeds
            if (std::abs(previousDiameter - otherTop_r) <
                filter_config.deltaRMin) {
                newCompSeed = false;
                break;
            }
        }

        if (newCompSeed) {
            compat_seedR[num_compat_seedR] = otherTop_r;
            triplet.weight += filter_config.compatSeedWeight;
            num_compat_seedR++;
        }

        if (num_compat_seedR >= filter_config.compatSeedLimit) {
            break;
        }
    }
}

}  // namespace cuda
}  // namespace traccc
