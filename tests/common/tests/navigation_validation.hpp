/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_parameters.hpp"

// Detray include(s).
#include <detray/navigation/intersection/intersection.hpp>

// Detray test include(s)
#include <detray/test/utils/inspectors.hpp>  //< candidate_record type

// System include(s)
#include <ranges>

namespace traccc::navigation_validator {

template <typename detector_t>
using intersection_type =
    detray::intersection2D<typename detector_t::surface_type,
                           typename detector_t::algebra_type, true>;

template <typename detector_t>
using candidate_type =
    detray::navigation::detail::candidate_record<intersection_type<detector_t>>;

/// Transcribe the hits for a particle to a candidate trace that detray requires
///
/// @param ctx the geometric context
/// @param det the detector
/// @param particle the truth particle
/// @param hits all hits in the event
/// @param n_hits_for_particle expected number of hits for the particle
///
/// @returns a vector of intersection candidate records
template <typename detector_t>
auto transcribe_to_trace(const typename detector_t::geometry_context ctx,
                         const detector_t& det,
                         const traccc::io::csv::particle& particle,
                         const std::vector<traccc::io::csv::hit>& hits,
                         const detray::navigation::direction nav_dir =
                             detray::navigation::direction::e_forward,
                         const std::size_t n_hits_for_particle = 10u) {
    using intersection_t =
        typename candidate_type<detector_t>::intersection_type;

    detray::dvector<candidate_type<detector_t>> candidates{};
    candidates.reserve(n_hits_for_particle);

    // Fill the hits into the candidate trace
    auto fill_trace = [&]<std::ranges::view hit_view_t>(hit_view_t hits_view) {
        for (const auto& h : hits_view) {
            if (h.particle_id != particle.particle_id) {
                continue;
            }

            // Rough estimate of path
            const point3 pos{h.tx, h.ty, h.tz};
            const vector3 p{h.tpx, h.tpy, h.tpz};
            const vector3 dir = vector::normalize(p);
            const scalar path{vector::norm(pos)};

            // Corresponding surface
            const detray::geometry::barcode bcd{h.geometry_id};
            const auto sf_desc = det.surfaces().at(bcd.index());
            const auto sf = detray::tracking_surface{det, bcd};

            // Build an intersection from the hit
            using nav_link_t = typename intersection_t::nav_link_t;
            auto loc_pos = sf.global_to_local(ctx, pos, dir);
            intersection_t intr{
                sf_desc, path, static_cast<nav_link_t>(bcd.volume()),
                true,    true, loc_pos};

            candidates.emplace_back(pos, dir, intr, particle.q,
                                    vector::norm(p));
        }
    };

    if (nav_dir == detray::navigation::direction::e_backward) {
        fill_trace(hits | std::views::reverse);
    } else {
        fill_trace(hits | std::views::all);
    }

    return candidates;
}

}  // namespace traccc::navigation_validator