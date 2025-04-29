/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "../../utils/barrier.hpp"
#include "../../utils/thread_id.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/geometry/detector.hpp"

namespace traccc::alpaka {

template <typename detector_t>
struct FindTracksKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const finding_config& cfg,
        device::find_tracks_payload<detector_t>* payload) const {

        auto& shared_num_out_params =
            ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        auto& shared_out_offset =
            ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        auto& shared_candidates_size =
            ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        unsigned long long int* const s =
            ::alpaka::getDynSharedMem<unsigned long long int>(acc);
        unsigned long long int* shared_insertion_mutex = s;

        alpaka::barrier<TAcc> barrier(&acc);
        details::thread_id1 thread_id(acc);

        unsigned int blockDimX = thread_id.getBlockDimX();
        std::pair<unsigned int, unsigned int>* shared_candidates =
            reinterpret_cast<std::pair<unsigned int, unsigned int>*>(
                &shared_insertion_mutex[blockDimX]);

        device::find_tracks<detector_t>(
            thread_id, barrier, cfg, *payload,
            {shared_num_out_params, shared_out_offset, shared_insertion_mutex,
             shared_candidates, shared_candidates_size});
    }
};

}  // namespace traccc::alpaka
