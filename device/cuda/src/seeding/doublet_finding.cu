/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/cuda/seeding/doublet_finding.hpp"
#include "traccc/cuda/utils/cuda_helper.cuh"
#include "traccc/cuda/utils/definitions.hpp"

namespace traccc {
namespace cuda {

__global__ void set_zero_kernel(doublet_container_view mbc_view,
                                doublet_container_view mtc_view) {
    device_doublet_container mbc_device({mbc_view.headers, mbc_view.items});
    mbc_device.get_headers().at(threadIdx.x).zeros();

    device_doublet_container mtc_device({mtc_view.headers, mtc_view.items});
    mtc_device.get_headers().at(threadIdx.x).zeros();
}

/// Forward declaration of doublet finding kernel
/// The mid-bot and mid-top doublets are found for the compatible middle
/// spacepoints which were recorded during doublet_counting
///
/// @param config seed finder config
/// @param internal_sp_view vecmem container for internal spacepoint
/// @param doublet_count_view vecmem container for doublet_counter
/// @param mid_bot_doublet_container vecmem container for mid-bot doublets
/// @param mid_top_doublet_container vecmem container for mid-top doublets
/// @param resource vecmem memory resource
__global__ void doublet_finding_kernel(
    const seedfinder_config config, sp_grid_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view);

void doublet_finding(const seedfinder_config& config,
                     const vecmem::vector<doublet_counter_per_bin>& dcc_headers,
                     sp_grid_view internal_sp_view,
                     doublet_counter_container_view dcc_view,
                     doublet_container_view mbc_view,
                     doublet_container_view mtc_view,
                     vecmem::memory_resource& resource) {

    unsigned int nbins = internal_sp_view._data_view.m_size;

    // zero initialization
    set_zero_kernel<<<1, nbins>>>(mbc_view, mtc_view);
    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // The thread-block is desinged to make each thread find doublets per
    // compatible middle spacepoints (comptible middle spacepoint means that the
    // number of mid-bot and mid-top doublets are larger than zero)

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    unsigned int num_threads = WARP_SIZE * 2;

    // -- Num blocks
    // The dimension of grid is = sum_i{N_i}, where:
    // i is the spacepoint bin index
    // N_i is the number of blocks for i-th bin, defined as
    // num_compatible_middle_sp_per_bin / num_threads + 1
    unsigned int num_blocks = 0;
    for (unsigned int i = 0; i < nbins; ++i) {
        num_blocks += dcc_headers[i].n_spM / num_threads + 1;
    }

    // shared memory assignment for the number of and mid_top doublets per
    // thread
    unsigned int sh_mem = sizeof(int) * num_threads * 2;

    // run the kernel
    doublet_finding_kernel<<<num_blocks, num_threads, sh_mem>>>(
        config, internal_sp_view, dcc_view, mbc_view, mtc_view);

    // cuda error check
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

__global__ void doublet_finding_kernel(
    const seedfinder_config config, sp_grid_view internal_sp_view,
    doublet_counter_container_view doublet_counter_view,
    doublet_container_view mid_bot_doublet_view,
    doublet_container_view mid_top_doublet_view) {

    // Get device container for input parameters
    sp_grid_device internal_sp_device(internal_sp_view);

    device_doublet_counter_container doublet_counter_device(
        {doublet_counter_view.headers, doublet_counter_view.items});

    device_doublet_container mid_bot_doublet_device(
        {mid_bot_doublet_view.headers, mid_bot_doublet_view.items});
    device_doublet_container mid_top_doublet_device(
        {mid_top_doublet_view.headers, mid_top_doublet_view.items});

    // Get the bin and item index
    unsigned int bin_idx(0), item_idx(0);
    cuda_helper::find_idx_on_container(doublet_counter_device, bin_idx,
                                       item_idx);

    // get internal spacepoints for current bin
    auto internal_sp_per_bin = internal_sp_device.bin(bin_idx);

    // Header of doublet counter : number of compatible middle sp per bin
    // Item of doublet counter : doublet counter objects per bin
    auto& num_compat_spM_per_bin =
        doublet_counter_device.get_headers().at(bin_idx).n_spM;

    auto doublet_counter_per_bin =
        doublet_counter_device.get_items().at(bin_idx);

    // Header of doublet: number of mid_bot doublets per bin
    // Item of doublet: doublet objects per bin
    auto& num_mid_bot_doublets_per_bin =
        mid_bot_doublet_device.get_headers().at(bin_idx).n_doublets;
    auto mid_bot_doublets_per_bin =
        mid_bot_doublet_device.get_items().at(bin_idx);

    // Header of doublet: number of mid_top doublets per bin
    // Item of doublet: doublet objects per bin
    auto& num_mid_top_doublets_per_bin =
        mid_top_doublet_device.get_headers().at(bin_idx).n_doublets;
    auto mid_top_doublets_per_bin =
        mid_top_doublet_device.get_items().at(bin_idx);

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

    // prevent the tail threads referring the null doublet counter
    if (item_idx >= num_compat_spM_per_bin) {
        return;
    }

    // index of internal spacepoint in the item vector
    auto sp_idx = doublet_counter_per_bin[item_idx].spM.sp_idx;
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
    for (unsigned int i = 0; i < item_idx; i++) {
        mid_bot_start_idx += doublet_counter_per_bin[i].n_mid_bot;
        mid_top_start_idx += doublet_counter_per_bin[i].n_mid_top;
    }

    // get the neighbor indices
    auto phi_bins =
        internal_sp_device.axis_p0().range(isp.phi(), config.neighbor_scope);
    auto z_bins =
        internal_sp_device.axis_p1().range(isp.z(), config.neighbor_scope);
    auto i_p = phi_bins[0];
    auto i_z = z_bins[0];

    for (unsigned int c = 0; c < config.get_max_neighbor_bins(); c++) {

        auto neighbors = internal_sp_device.bin(i_p, i_z);
        auto neigh_bin = static_cast<unsigned int>(
            i_p + i_z * internal_sp_device.axis_p0().bins());

        // for (auto& neigh_isp: neighbors){
        for (unsigned int spB_idx = 0; spB_idx < neighbors.size(); spB_idx++) {
            const auto& neigh_isp = neighbors[spB_idx];

            // Check if middle and bottom sp can form a doublet
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                     true)) {
                auto spB_loc = sp_location({neigh_bin, spB_idx});

                // Check conditions
                // 1) number of mid-bot doublets per spM should be smaller than
                // what is counted in doublet_counting (so it should be true
                // always) 2) prevent overflow
                if (n_mid_bot_per_spM <
                        doublet_counter_per_bin[item_idx].n_mid_bot &&
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
            if (doublet_finding_helper::isCompatible(isp, neigh_isp, config,
                                                     false)) {
                auto spT_loc = sp_location({neigh_bin, spB_idx});

                // Check conditions
                // 1) number of mid-top doublets per spM should be smaller than
                // what is counted in doublet_counting (so it should be true
                // always) 2) prevent overflow
                if (n_mid_top_per_spM <
                        doublet_counter_per_bin[item_idx].n_mid_top &&
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

        i_z++;
        // terminate if we went through all neighbor bins
        if (i_z > z_bins[1] && i_p == phi_bins[1]) {
            break;
        }

        // increse phi_index and reset i_z
        // if i_z is larger than last neighbor index
        if (i_z > z_bins[1]) {
            i_p++;
            i_p = i_p % internal_sp_device.axis_p0().bins();
            i_z = z_bins[0];
        }
    }

    // Calculate the number doublets per "block" with reducing sum technique
    __syncthreads();
    cuda_helper::reduce_sum<int>(num_mid_bot_doublets_per_thread);
    __syncthreads();
    cuda_helper::reduce_sum<int>(num_mid_top_doublets_per_thread);

    // Calculate the number doublets per bin by atomic-adding the number of
    // doublets per block
    if (threadIdx.x == 0) {
        atomicAdd(&num_mid_bot_doublets_per_bin,
                  num_mid_bot_doublets_per_thread[0]);
        atomicAdd(&num_mid_top_doublets_per_bin,
                  num_mid_top_doublets_per_thread[0]);
    }
}

}  // namespace cuda
}  // namespace traccc
