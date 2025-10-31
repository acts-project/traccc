/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/detail/track_params_estimation_config.hpp"
#include "traccc/seeding/device/estimate_track_params.hpp"
#include "traccc/seeding/track_params_estimation_helper.hpp"

// System include(s).
#include <cassert>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void estimate_track_params(
    const global_index_t globalIndex,
    const track_params_estimation_config& config,
    const measurement_collection_types::const_view& measurements_view,
    const edm::spacepoint_collection::const_view& spacepoints_view,
    const edm::seed_collection::const_view& seeds_view, const vector3& bfield,
    bound_track_parameters_collection_types::view params_view) {

    // Check if anything needs to be done.
    const edm::seed_collection::const_device seeds_device(seeds_view);
    if (globalIndex >= seeds_device.size()) {
        return;
    }

    const measurement_collection_types::const_device measurements_device(
        measurements_view);
    const edm::spacepoint_collection::const_device spacepoints_device(
        spacepoints_view);

    bound_track_parameters_collection_types::device params_device(params_view);

    const edm::seed_collection::const_device::const_proxy_type this_seed =
        seeds_device.at(globalIndex);

    // Get bound track parameter
    bound_track_parameters<>& track_params = params_device.at(globalIndex);
    seed_to_bound_param_vector(track_params, measurements_device,
                               spacepoints_device, this_seed, bfield);

    // NOTE: The code below uses the covariance of theta in the calculation of
    // the calculation of q/p. Thus, theta must be computed first.
    static_assert(e_bound_qoverp > e_bound_theta);

    // Set Covariance
    for (std::size_t i = 0; i < e_bound_size; i++) {
        scalar var = config.initial_sigma[i] * config.initial_sigma[i];

        if (i == e_bound_qoverp) {
            const scalar var_theta = getter::element(
                track_params.covariance(), e_bound_theta, e_bound_theta);

            // Contribution from sigma(q/pt)
            const scalar sigma_qopt = config.initial_sigma_qopt *
                                      math::sin(track_params[e_bound_theta]);
            var += sigma_qopt * sigma_qopt;

            // Contribution from sigma(pt)/pt
            const scalar sigma_pt_rel =
                config.initial_sigma_pt_rel * track_params[e_bound_qoverp];
            var += sigma_pt_rel * sigma_pt_rel;

            // Contribution from sigma(theta)
            scalar sigma_theta = track_params[e_bound_qoverp] /
                                 math::tan(track_params[e_bound_theta]);
            var += var_theta * sigma_theta * sigma_theta;
        }

        var *= config.initial_inflation[i];

        getter::element(track_params.covariance(), i, i) = var;
    }
}

}  // namespace traccc::device
