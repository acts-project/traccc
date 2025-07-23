/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/particle.hpp"
#include "traccc/edm/track_parameters.hpp"

// Detray include(s).
#include <detray/navigation/intersection/intersection.hpp>

// Detray test include(s)
#include <detray/test/utils/inspectors.hpp>  //< candidate_record type

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
                         const scalar min_p = 50.f * traccc::unit<scalar>::MeV,
                         const scalar max_rad = 20.f * traccc::unit<scalar>::mm,
                         const std::size_t n_hits_for_particle = 10u) {
    using intersection_t =
        typename candidate_type<detector_t>::intersection_type;

    detray::dvector<candidate_type<detector_t>> candidates{};

    // Apply momentum cut to get rid of some secondaries
    if (vector::norm(vector3{particle.px, particle.py, particle.pz}) < min_p) {
        return candidates;
    }

    const scalar q{particle.q};

    candidates.reserve(n_hits_for_particle);

    // Fill the hits into the candidate trace
    for (const auto& h : hits) {
        if (h.particle_id != particle.particle_id) {
            continue;
        }

        // Rough estimate of path
        const point3 pos{h.tx, h.ty, h.tz};
        const vector3 mom{h.tpx, h.tpy, h.tpz};
        const vector3 dir = vector::normalize(mom);
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

        candidates.emplace_back(pos, dir, intr, q, vector::norm(mom));
    }

    // Remove secondaries
    if (vector::perp(candidates.front().pos) > max_rad) {
        return detray::dvector<candidate_type<detector_t>>{};
    }

    return candidates;
}

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
auto transcribe_to_trace(
    const typename detector_t::geometry_context ctx, const detector_t& det,
    const traccc::particle& ptc,
    const std::map<traccc::particle, std::vector<traccc::measurement>>&
        ptc_to_meas_map,
    const scalar min_p = 50.f * traccc::unit<scalar>::MeV,
    const scalar max_rad = 50.f * traccc::unit<scalar>::mm,
    const std::size_t n_meas_for_particle = 10u) {

    using intersection_t =
        typename candidate_type<detector_t>::intersection_type;

    detray::dvector<candidate_type<detector_t>> candidates{};

    // TODO: Not accurate for every measurement
    const scalar p{vector::norm(ptc.momentum)};
    const scalar q{ptc.charge};

    // Check if particle produced any hits or has too small momentum
    if (!ptc_to_meas_map.contains(ptc) || p < min_p) {
        return candidates;
    }

    candidates.reserve(n_meas_for_particle);

    // Fill the hits into the candidate trace
    for (const auto& meas : ptc_to_meas_map.at(ptc)) {
        // Corresponding surface
        const detray::geometry::barcode bcd{meas.surface_link};
        const auto sf_desc = det.surfaces().at(bcd.index());
        const auto sf = detray::tracking_surface{det, bcd};

        // TODO: Use correct track direction at measurement for line sf.
        const vector3 dir{vector::normalize(ptc.momentum)};
        const point3 glob_pos{sf.local_to_global(ctx, meas.local, dir)};

        // Rough estimate of intersection distance from origin
        const scalar path{vector::norm(glob_pos)};

        // Build an intersection
        using nav_link_t = typename intersection_t::nav_link_t;
        intersection_t intr{
            {sf_desc, path, static_cast<nav_link_t>(bcd.volume()), true, true},
            {meas.local[0], meas.local[1], 0.f}};

        // TODO: Don't use intial particle momentum
        candidates.emplace_back(glob_pos, dir, intr, q, p);
    }

    // Remove secondaries
    if (vector::perp(candidates.front().pos) > max_rad) {
        return detray::dvector<candidate_type<detector_t>>{};
    }

    return candidates;
}

}  // namespace traccc::navigation_validator
