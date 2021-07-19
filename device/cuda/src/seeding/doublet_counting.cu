/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda/seeding/doublet_counting.cuh>
#include <cuda/utils/cuda_helper.cuh>
#include <cuda/utils/definitions.hpp>

namespace traccc {
namespace cuda {

/// Forward declaration of doublet counting kernel
/// The number of mid-bot and mid-top doublets are counted for all spacepoints and recorded into doublet counter container if the number of doublets are larger than zero
/// 
/// @param config seed finder config
/// @param internal_sp_view vecmem container for internal spacepoint
/// @param doublet_count_view vecmem container for doublet_counter
__global__ void doublet_counting_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_count_view);

void doublet_counting(const seedfinder_config& config,
                      host_internal_spacepoint_container& internal_sp_container,
                      host_doublet_counter_container& doublet_counter_container,
                      vecmem::memory_resource* resource) {
    auto internal_sp_view = get_data(internal_sp_container, resource);
    auto doublet_counter_container_view =
        get_data(doublet_counter_container, resource);

    // The thread-block is desinged to make each thread count the number of doublets per middle spacepoint 

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks 
    // The dimension of grid is = sum_i{N_i}, where:
    // i is the spacepoint bin index
    // N_i is the number of blocks for i-th bin, defined as num_middle_sp_per_bin / num_threads + 1  
    unsigned int num_blocks = 0;
    for (size_t i = 0; i < internal_sp_view.headers.size(); ++i) {
        num_blocks += internal_sp_view.items.m_ptr[i].size() / num_threads + 1;
    }

    // run the kernel
    doublet_counting_kernel<<<num_blocks, num_threads>>>(
        config, internal_sp_view, doublet_counter_container_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
    
__global__ void doublet_counting_kernel(
    const seedfinder_config config,
    internal_spacepoint_container_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view) {

    // Get device container for input parameters
    device_internal_spacepoint_container internal_sp_device(
        {internal_sp_view.headers, internal_sp_view.items});
    device_doublet_counter_container doublet_counter_device(
        {doublet_counter_view.headers, doublet_counter_view.items});

    // Get the bin index of spacepoint binning and reference block idx for the bin index   
    unsigned int bin_idx = 0;
    unsigned int ref_block_idx = 0;
    cuda_helper::get_header_idx(internal_sp_device.items, bin_idx,
                             ref_block_idx);

    // Header of internal spacepoint container : spacepoint bin information
    // Item of internal spacepoint container : internal spacepoint objects per bin
    const auto& bin_info = internal_sp_device.headers.at(bin_idx);
    auto internal_sp_per_bin = internal_sp_device.items.at(bin_idx);

    // Header of doublet counter : number of compatible middle sp per bin
    // Item of doublet counter : doublet counter objects per bin
    auto& num_compat_spM_per_bin = doublet_counter_device.headers.at(bin_idx);
    auto doublet_counter_per_bin = doublet_counter_device.items.at(bin_idx);

    // index of internal spacepoint in the item vector
    auto sp_idx = (blockIdx.x - ref_block_idx) * blockDim.x + threadIdx.x;

    // kill the process before overflow
    if (sp_idx >= doublet_counter_per_bin.size()) {
        return;
    }

    // zero initialization for the number of doublets per thread (or middle sp)
    unsigned int n_mid_bot = 0;
    unsigned int n_mid_top = 0;

    // zero initialization for the number of doublets per bin
    doublet_counter_per_bin[sp_idx].n_mid_bot = 0;
    doublet_counter_per_bin[sp_idx].n_mid_top = 0;

    // middle spacepoint index
    auto spM_loc = sp_location({bin_idx, sp_idx}); 
    // middle spacepoint
    const auto& isp = internal_sp_per_bin[sp_idx]; 

    // Loop over (bottom and top) internal spacepoints in the neighbor bins
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
                n_mid_bot++;
            }

	    // Check if middle and top sp can form a doublet
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                     false)) {
                n_mid_top++;
            }
        }
    }

    // if number of mid-bot and mid-top doublet for a middle spacepoint is larger than 0, the entry is added to the doublet counter
    if (n_mid_bot > 0 && n_mid_top > 0) {
        auto pos = atomicAdd(&num_compat_spM_per_bin, 1);
        doublet_counter_per_bin[pos].spM = spM_loc;
        doublet_counter_per_bin[pos].n_mid_bot = n_mid_bot;
        doublet_counter_per_bin[pos].n_mid_top = n_mid_top;
    }
}

}  // namespace cuda
}  // namespace traccc
