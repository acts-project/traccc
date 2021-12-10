/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "seeding/track_params_estimation_helper.hpp"
#include "utils/algorithm.hpp"

namespace traccc {

/// track parameters estimation
/// Originated from Acts/Seeding/EstimateTrackParamsFromSeed.hpp

struct track_params_estimation
    : public algorithm<host_bound_track_parameters_collection(
          const host_seed_container&)> {
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
    output_type operator()(const host_seed_container& i) const override {
        output_type result(&m_mr.get());
        this->operator()(i, result);
        return result;
    }

    /// Callable operator for track_params_esitmation
    ///
    /// @param input_type is the seed container
    ///
    /// void interface
    ///
    /// @return vector of bound track parameter
    void operator()(const host_seed_container& i, output_type& o) const {
        // input for seed vector
        const auto& seeds = i.get_items()[0];

        // convenient assumption on bfield and mass
        // TODO: Make use of bfield extenstion in the future
        vector3 bfield = {0, 0, 2};

        for (auto seed : seeds) {
            bound_track_parameters track_params;
            track_params.vector() =
                seed_to_bound_vector(seed, bfield, PION_MASS_MEV);

            o.push_back(track_params);
        }
    }

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc
