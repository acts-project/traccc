/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "traccc/cuda/utils/cuda_helper.hpp"
#include "traccc/kokkos/seeding/triplet_counting.hpp"
#include "traccc/kokkos/utils/kokkos_helper.hpp"

namespace traccc {

// defining MemSpace to generalize in case they use other
#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#endif
    
#ifndef MemSpace
#define MemSpace Kokkos::HostSpace
#endif 

// defining execution space and range_policy
using ExecSpace = Memspace::execution_space;
using range_policy = Kokkos::RangePolicy<ExecSpace>; 
 
typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
typedef Kokkos::TeamPolicy<ExecSpace>::member_type member_type;

namespace kokkos {

void triplet_counting(
    const seedfinder_config& config,
    const vecmem::vector<doublet_per_bin>& mbc_headers,
    sp_grid_const_view internal_sp_view,
    device::doublet_counter_container_types::const_view dcc_view,
    doublet_container_view mbc_view, doublet_container_view mtc_view,
    triplet_counter_container_view tcc_view,
    vecmem::memory_resource& resource) {

    unsigned int nbins = internal_sp_view._data_view.m_size;
    
    unsigned int num_threads = WARP_SIZE * 2;
    unsigned int num_blocks = nbins / num_threads + 1;

    Kokkos::parallel_for("set_zero_kernel", team_policy(num_blocks, num_threads),
        KOKKOS_LAMBDA (const member_type &team_member) {
            const std::size_t gid = 
                team_member.league_rank() * team_member.team_size() + team_member.team_rank();
            device_triplet_counter_container tcc_device(tcc_view);

            if (gid >= tcc_device.get_headers().size()) {
                return;
            }

            tcc_device.get_headers().at(gid).zeros();
        }
    );

    // The thread-block is desinged to make each thread count triplets per
    // middle-bot doublet

    // -- Num threads
    // The dimension of block is the integer multiple of WARP_SIZE (=32)
    num_threads = WARP_SIZE * 4;

    // -- Num blocks
    // The dimension of grid is = sum_i{N_i}, where:
    // i is the spacepoint bin index
    // N_i is the number of blocks for i-th bin, defined as
    // num_mid_bot_doublet_per_bin / num_threads + 1
    num_blocks = 0;
    for (size_t i = 0; i < nbins; ++i) {
        num_blocks += mbc_headers[i].n_doublets / num_threads + 1;
    }

    Kokkos::parallel_for("triplet_counting", team_policy(num_blocks, num_threads),
        KOKKOS_LAMBDA (const member_type &team_member) {
          // for what i understand all devices MUST have the same size
          // Get device container for input parameters
          const_sp_grid_device internal_sp_device(internal_sp_view);

          const device::doublet_counter_container_types::const_device
              doublet_counter_device(dcc_view);
          device_doublet_container mid_bot_doublet_device(mbc_view);
          device_doublet_container mid_top_doublet_device(mtc_view);
          device_triplet_counter_container triplet_counter_device(
              tcc_view);

          // Get the bin and item index
          // based of the kokkos logic the index is the same value
          unsigned int bin_idx(0), mb_idx(0);
          kokkos_helper::find_idx_on_container(mid_bot_double_device, 
                                               team_member.team_size(),
                                               team_member.league_rank(),
                                               team_member.team_rank(),
                                               bin_idx, mb_idx);

          // get internal spacepoints for current bin
          auto internal_sp_per_bin = internal_sp_device.bin(bin_idx);
          auto& num_compat_spM_per_bin =
              doublet_counter_device.get_headers().at(bin_idx).m_nSpM;

          // Header of doublet counter : number of compatible middle sp per bin
          // Item of doublet counter : doublet counter objects per bin
          auto doublet_counter_per_bin =
              doublet_counter_device.get_items().at(bin_idx);

          // Header of doublet: number of mid_bot doublets per bin
          // Item of doublet: doublet objects per bin
          const auto& num_mid_bot_doublets_per_bin =
              mid_bot_doublet_device.get_headers().at(bin_idx).n_doublets;
          auto mid_bot_doublets_per_bin =
              mid_bot_doublet_device.get_items().at(bin_idx);

          // Header of doublet: number of mid_top doublets per bin
          // Item of doublet: doublet objects per bin
          const auto& num_mid_top_doublets_per_bin =
              mid_top_doublet_device.get_headers().at(bin_idx).n_doublets;
          auto mid_top_doublets_per_bin =
              mid_top_doublet_device.get_items().at(bin_idx);
 
          // Header of triplet counter: number of compatible mid_top doublets per bin
          // Item of triplet counter: triplet counter objects per bin
          auto& num_compat_mb_per_bin =
              triplet_counter_device.get_headers().at(bin_idx).n_mid_bot;
          auto& num_triplets =
              triplet_counter_device.get_headers().at(bin_idx).n_triplets;
 
          auto triplet_counter_per_bin =
              triplet_counter_device.get_items().at(bin_idx);
 
          // prevent the tail threads referring the null doublet counter
          if (mb_idx >= num_mid_bot_doublets_per_bin) {
              return;
          }
 
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
          const auto& spB = internal_sp_device.bin(spB_bin)[spB_idx];
 
          // Apply the conformal transformation to middle-bot doublet
          auto lb = doublet_finding_helper::transform_coordinates(spM, spB, true);
 
          // Calculate some physical quantities required for triplet compatibility
          // check
          scalar iSinTheta2 = 1 + lb.cotTheta() * lb.cotTheta();
          scalar scatteringInRegion2 = config.maxScatteringAngle2 * iSinTheta2;
          scatteringInRegion2 *= config.sigmaScattering * config.sigmaScattering;
          scalar curvature, impact_parameter;
 
          // find the reference (start) index of the mid-top doublet container item
          // vector, where the doublets are recorded The start index is calculated by
          // accumulating the number of mid-top doublets of all previous compatible
          // middle spacepoints
          unsigned int mb_end_idx = 0;
          unsigned int mt_start_idx = 0;
          unsigned int mt_end_idx = 0;
 
          for (unsigned int i = 0; i < num_compat_spM_per_bin; ++i) {
              mb_end_idx += doublet_counter_per_bin[i].m_nMidBot;
              mt_end_idx += doublet_counter_per_bin[i].m_nMidTop;
 
              if (mb_end_idx > mb_idx) {
                  break;
              }
              mt_start_idx += doublet_counter_per_bin[i].m_nMidTop;
          }
 
          if (mt_end_idx >= mid_top_doublets_per_bin.size()) {
              mt_end_idx = fmin(mid_top_doublets_per_bin.size(), mt_end_idx);
          }
 
          if (mt_start_idx >= mid_top_doublets_per_bin.size()) {
              return;
          }
 
          // number of triplets per thread (or per middle-bot doublet)
          unsigned int num_triplets_per_mb = 0;
 
          // iterate over mid-top doublets
          for (unsigned int i = mt_start_idx; i < mt_end_idx; ++i) {
              const auto& mid_top_doublet = mid_top_doublets_per_bin[i];
 
              const auto& spT_bin = mid_top_doublet.sp2.bin_idx;
              const auto& spT_idx = mid_top_doublet.sp2.sp_idx;
              const auto& spT = internal_sp_device.bin(spT_bin)[spT_idx];
 
              // Apply the conformal transformation to middle-top doublet
              auto lt =
                  doublet_finding_helper::transform_coordinates(spM, spT, false);
 
              // Check if mid-bot and mid-top doublets can form a triplet
              if (triplet_finding_helper::isCompatible(
                      spM, lb, lt, config, iSinTheta2, scatteringInRegion2, curvature,
                      impact_parameter)) {
                  num_triplets_per_mb++;
              }
          }
 
          // if the number of triplets per mb is larger than 0, write the triplet
          // counter into the container
          if (num_triplets_per_mb > 0) {
              auto pos = Kokkos::atomic_add(&num_compat_mb_per_bin, 1);
              Kokkos::atomic_add(&num_triplets, num_triplets_per_mb);
              triplet_counter_per_bin[pos] = {mid_bot_doublet, num_triplets_per_mb};
          } 
        }
    )
}


}  // namespace kokkos
}  // namespace traccc
