/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/device/estimate_track_params.hpp"
#include "traccc/seeding/track_params_estimation_helper.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void estimate_track_params(
    const std::size_t globalIndex,
    const spacepoint_container_types::const_view& spacepoints_view,
    const seed_collection_types::const_view& seeds_view,
    bound_track_parameters_collection_types::view params_view) {

    // Check if anything needs to be done.
    const seed_collection_types::const_device seeds_device(seeds_view);
    if (globalIndex >= seeds_device.size()) {
        return;
    }

    const spacepoint_container_types::const_device spacepoints_device(
        spacepoints_view);

    bound_track_parameters_collection_types::device params_device(params_view);

    // convenient assumption on bfield and mass
    vector3 bfield = {0, 0, 2};

    const seed& this_seed = seeds_device.at(globalIndex);

    // Get bound track parameter
    const auto param = seed_to_bound_vector(spacepoints_device, this_seed,
                                            bfield, PION_MASS_MEV);
    params_device[globalIndex].set_vector(param);
}

}  // namespace traccc::device
