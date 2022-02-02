/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/seeding/track_params_estimation_helper.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc {

/// track parameters estimation
/// Originated from Acts/Seeding/EstimateTrackParamsFromSeed.hpp

struct track_params_estimation
    : public algorithm<host_bound_track_parameters_collection(
          const host_spacepoint_container&, const host_seed_collection&)> {
    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    track_params_estimation(vecmem::memory_resource& mr) : m_mr(mr) {}

    /// Callable operator for track_params_esitmation
    ///
    /// @param input_type is the seed container
    ///
    /// @return vector of bound track parameters
    output_type operator()(const host_spacepoint_container& sp_container,
                           const host_seed_collection& seeds) const override {
        output_type result(&m_mr.get());

        // convenient assumption on bfield and mass
        // TODO: Make use of bfield extenstion in the future
        vector3 bfield = {0, 0, 2};

        for (auto seed : seeds) {
            bound_track_parameters track_params;
            track_params.vector() =
                seed_to_bound_vector(sp_container, seed, bfield, PION_MASS_MEV);

            result.push_back(track_params);
        }

        return result;
    }

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc
