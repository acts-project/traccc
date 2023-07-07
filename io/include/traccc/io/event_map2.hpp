/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_candidate.hpp"

namespace traccc {

/// Measurement with surface link
struct measurement_link {
    geometry_id surface_link;
    measurement meas;
};

/// Comparison / ordering operator for measurements
TRACCC_HOST_DEVICE
inline bool operator<(const measurement_link& lhs,
                      const measurement_link& rhs) {
    if (lhs.surface_link < rhs.surface_link) {
        return true;
    } else if (lhs.surface_link == rhs.surface_link) {
        return lhs.meas < rhs.meas;
    }

    return false;
}

struct event_map2 {

    /// Constructor without cell information
    event_map2(std::size_t event, const std::string& measurement_dir,
               const std::string& hit_dir, const std::string particle_dir);

    template <typename seed_generator_t>
    track_candidate_container_types::host generate_truth_candidates(
        seed_generator_t& sg, vecmem::memory_resource& resource) {

        traccc::track_candidate_container_types::host track_candidates(
            &resource);

        for (auto const& [ptc, measurements] : ptc_meas_map) {

            const auto& xp = meas_xp_map[measurements[0]];
            const free_track_parameters free_param(xp.first, 0.f, xp.second,
                                                   ptc.charge);

            auto seed_params = sg(measurements[0].surface_link, free_param);

            // Candidate objects
            vecmem::vector<track_candidate> candidates;

            for (const auto& meas_link : measurements) {

                track_candidate cand = {
                    detray::geometry::barcode{meas_link.surface_link},
                    meas_link.meas};

                candidates.push_back(cand);
            }

            track_candidates.push_back(std::move(seed_params),
                                       std::move(candidates));
        }

        return track_candidates;
    }

    /// Map for measurement to truth global position and momentum
    using measurement_xp_map =
        std::map<measurement_link, std::pair<point3, point3>>;
    /// Map for measurement to the contributing particles
    using measurement_particle_map =
        std::map<measurement_link, std::map<particle, uint64_t>>;
    /// Map for particle to the vector of (geometry_id, measurement)
    using particle_measurement_map =
        std::map<particle, std::vector<measurement_link>>;
    using particle_id = uint64_t;
    using particle_map = std::map<particle_id, particle>;

    particle_map ptc_map;
    measurement_xp_map meas_xp_map;
    measurement_particle_map meas_ptc_map;
    particle_measurement_map ptc_meas_map;
};

}  // namespace traccc