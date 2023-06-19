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
    const seed_collection_types::host& seeds, const vector3& bfield) const {

    const unsigned int num_seeds = seeds.size();
    output_type result(num_seeds, &m_mr.get());

    for (unsigned int i = 0; i < num_seeds; ++i) {
        bound_track_parameters track_params;
        track_params.set_vector(
            seed_to_bound_vector(spacepoints, seeds[i], bfield, PION_MASS_MEV));

        result[i] = track_params;
    }

    return result;
}

}  // namespace traccc
