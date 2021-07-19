/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/seeding/doublet_finding.cuh>
#include <cuda/utils/cuda_helper.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

/// Forward declaration of doublet finding kernel
/// The mid-bot and mid-top doublets are found for the compatible middle spacepoints which were recorded during doublet_counting
///    
/// @param config seed finder config
/// @param internal_sp_view vecmem container for internal spacepoint
/// @param doublet_count_view vecmem container for doublet_counter
/// @param mid_bot_doublet_container vecmem container for mid-bot doublets
/// @param mid_top_doublet_container vecmem container for mid-top doublets
/// @param resource vecmem memory resource        
__global__ void doublet_finding_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_data,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view);

void doublet_finding(const seedfinder_config& config,
                     host_internal_spacepoint_container& internal_sp_container,
                     host_doublet_counter_container& doublet_counter_container,
                     host_doublet_container& mid_bot_doublet_container,
                     host_doublet_container& mid_top_doublet_container,
                     vecmem::memory_resource* resource) {
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_view = get_data(doublet_counter_container, resource);
    auto mid_bot_doublet_view = get_data(mid_bot_doublet_container, resource);
    auto mid_top_doublet_view = get_data(mid_top_doublet_container, resource);

    // The thread-block is desinged to make each thread find doublets per compatible middle spacepoints (comptible middle spacepoint means that the number of mid-bot and mid-top doublets are larger than zero)

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)    
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks 
    // The dimension of grid is = sum_i{N_i}, where:
    // i is the spacepoint bin index
    // N_i is the number of blocks for i-th bin, defined as num_compatible_middle_sp_per_bin / num_threads + 1
    unsigned int num_blocks = 0;
    for (size_t i = 0; i < internal_sp_view.headers.size(); ++i) {
        num_blocks += doublet_counter_container.headers[i] / num_threads + 1;
    }

    // shared memory assignment for the number of and mid_top doublets per thread
    unsigned int sh_mem = sizeof(int) * num_threads * 2;

    // run the kernel    
    doublet_finding_kernel<<<num_blocks, num_threads, sh_mem>>>(
        config, internal_sp_view, doublet_counter_view, mid_bot_doublet_view,
        mid_top_doublet_view);

    // cuda error check    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void doublet_finding_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view) {
    device_internal_spacepoint_container internal_sp_device(
        {internal_sp_view.headers, internal_sp_view.items});
    device_doublet_counter_container doublet_counter_device(
        {doublet_counter_view.headers, doublet_counter_view.items});

    device_doublet_container mid_bot_doublet_device(
        {mid_bot_doublet_view.headers, mid_bot_doublet_view.items});
    device_doublet_container mid_top_doublet_device(
        {mid_top_doublet_view.headers, mid_top_doublet_view.items});

    // Get the bin index of spacepoint binning and reference block idx for the bin index       
    unsigned int bin_idx = 0;
    unsigned int ref_block_idx = 0;
    cuda_helper::get_header_idx(doublet_counter_device, bin_idx,
                             ref_block_idx);

    // Header of internal spacepoint container : spacepoint bin information
    // Item of internal spacepoint container : internal spacepoint objects per bin
    const auto& bin_info = internal_sp_device.headers.at(bin_idx);
    auto internal_sp_per_bin = internal_sp_device.items.at(bin_idx);

    // Header of doublet counter : number of compatible middle sp per bin
    // Item of doublet counter : doublet counter objects per bin    
    auto& num_compat_spM_per_bin = doublet_counter_device.headers.at(bin_idx);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(bin_idx);

    // Header of doublet: number of mid_bot doublets per bin
    // Item of doublet: doublet objects per bin    
    auto& num_mid_bot_doublets_per_bin =
        mid_bot_doublet_device.headers.at(bin_idx);
    auto mid_bot_doublets_per_bin = mid_bot_doublet_device.items.at(bin_idx);

    // Header of doublet: number of mid_top doublets per bin
    // Item of doublet: doublet objects per bin        
    auto& num_mid_top_doublets_per_bin =
        mid_top_doublet_device.headers.at(bin_idx);
    auto mid_top_doublets_per_bin = mid_top_doublet_device.items.at(bin_idx);

    // zero initialization for the number of doublets per threads
    extern __shared__ int num_doublets_per_thread[];
    int* num_mid_bot_doublets_per_thread = num_doublets_per_thread;
    int* num_mid_top_doublets_per_thread =
        &num_mid_bot_doublets_per_thread[blockDim.x];
    num_mid_bot_doublets_per_thread[threadIdx.x] = 0;
    num_mid_top_doublets_per_thread[threadIdx.x] = 0;    

    // Convenient alias for the number of doublets per thread
    auto& n_mid_bot_per_spM = num_mid_bot_doublets_per_thread[threadIdx.x];
    auto& n_mid_top_per_spM = num_mid_top_doublets_per_thread[threadIdx.x];
        
    // index of doublet counter in the item vector    
    auto gid = (blockIdx.x - ref_block_idx) * blockDim.x + threadIdx.x;

    // prevent the tail threads referring the null doublet counter
    if (gid >= num_compat_spM_per_bin) {
        return;
    }

    // index of internal spacepoint in the item vector    
    auto sp_idx = doublet_counter_per_bin[gid].spM.sp_idx;    
    // middle spacepoint index
    auto spM_loc = sp_location({bin_idx, sp_idx});
    // middle spacepoint
    auto& isp = internal_sp_per_bin[sp_idx];

    // find the reference (start) index of the doublet container item vector, where the doublets are recorded
    // The start index is calculated by accumulating the number of doublets of all previous compatible middle spacepoints
    unsigned int mid_bot_start_idx = 0;
    unsigned int mid_top_start_idx = 0;
    for (size_t i = 0; i < gid; i++) {
        mid_bot_start_idx += doublet_counter_per_bin[i].n_mid_bot;
        mid_top_start_idx += doublet_counter_per_bin[i].n_mid_top;
    }

    // Loop over (bottom and top) internal spacepoints in tje neighbor bins    
    for (size_t i_n = 0; i_n < bin_info.bottom_idx.counts; ++i_n) {
        const auto& neigh_bin = bin_info.bottom_idx.vector_indices[i_n];
        const auto& neigh_internal_sp_per_bin =
            internal_sp_device.items.at(neigh_bin);

        for (size_t spB_idx = 0; spB_idx < neigh_internal_sp_per_bin.size();
             ++spB_idx) {
            const auto& neigh_isp = neigh_internal_sp_per_bin[spB_idx];

	    // Check if middle and bottom sp can form a doublet
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                     true)) {
                auto spB_loc = sp_location({neigh_bin, spB_idx});

		// Check conditions
		// 1) number of mid-bot doublets per spM should be smaller than what is counted in doublet_counting (so it should be true always)
		// 2) prevent overflow
                if (n_mid_bot_per_spM <
                        doublet_counter_per_bin[gid].n_mid_bot &&
                    num_mid_bot_doublets_per_bin <
                        mid_bot_doublets_per_bin.size()) {
                    size_t pos = mid_bot_start_idx + n_mid_bot_per_spM;

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
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                     false)) {
                auto spT_loc = sp_location({neigh_bin, spB_idx});

		// Check conditions
		// 1) number of mid-top doublets per spM should be smaller than what is counted in doublet_counting (so it should be true always)
		// 2) prevent overflow		
                if (n_mid_top_per_spM <
                        doublet_counter_per_bin[gid].n_mid_top &&
                    num_mid_top_doublets_per_bin <
                        mid_top_doublets_per_bin.size()) {
                    size_t pos = mid_top_start_idx + n_mid_top_per_spM;

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
    __syncthreads();
    cuda_helper::reduce_sum<int>(num_mid_bot_doublets_per_thread);
    __syncthreads();
    cuda_helper::reduce_sum<int>(num_mid_top_doublets_per_thread);

    // Calculate the number doublets per bin by atomic-adding the number of doublets per block
    if (threadIdx.x == 0) {
        atomicAdd(&num_mid_bot_doublets_per_bin,
                  num_mid_bot_doublets_per_thread[0]);
        atomicAdd(&num_mid_top_doublets_per_bin,
                  num_mid_top_doublets_per_thread[0]);
    }
}

}  // namespace cuda
}  // namespace traccc
