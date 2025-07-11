/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "reset_status.cuh"

namespace traccc::cuda::kernels {

__global__ void reset_status(device::reset_status_payload payload) {

    // If it is the first iteration, do not terminate
    if (*(payload.is_first_iteration) == 1) {
        *(payload.terminate) = 0;
        *(payload.is_first_iteration) = 0;
    } else {
        // If the max number of shared measurements or the number of remaninig
        // tracks are zero, terminate
        if (*(payload.max_shared) == 0 || *(payload.n_accepted) == 0) {
            *(payload.terminate) = 1;
        }
    }

    // Reset the max number of shared measurements and the number of
    // updated tracks
    if (*(payload.terminate) == 0) {
        *(payload.max_shared) = 0;
        *(payload.n_updated_tracks) = 0;
    }
}

}  // namespace traccc::cuda::kernels
