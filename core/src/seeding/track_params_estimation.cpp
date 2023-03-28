/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/track_params_estimation.hpp"

#include "traccc/seeding/track_params_estimation_helper.hpp"

namespace traccc {

track_params_estimation::track_params_estimation(vecmem::memory_resource& mr)
    : m_mr(mr) {}

track_params_estimation::output_type track_params_estimation::operator()(
    const spacepoint_collection_types::host& spacepoints,
    const seed_collection_types::host& seeds) const {

    output_type result(&m_mr.get());

    // convenient assumption on bfield and mass
    // TODO: Make use of bfield extenstion in the future
    vector3 bfield = {0, 0, 2};

    for (const auto& seed : seeds) {
        bound_track_parameters track_params;
        track_params.set_vector(
            seed_to_bound_vector(spacepoints, seed, bfield, PION_MASS_MEV));

        result.push_back(track_params);
    }

    return result;
}

}  // namespace traccc
