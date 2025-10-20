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

// System include(s)
#include <algorithm>

namespace traccc::propagation_validator {

template <typename detector_t>
using intersection_type =
    detray::intersection2D<typename detector_t::surface_type,
                           typename detector_t::algebra_type, true>;

template <typename detector_t>
using candidate_type =
    detray::navigation::detail::candidate_record<intersection_type<detector_t>>;

/// Transcribe the hits for a particle to a candidate trace for the detray
/// propagation validation tools
///
/// @param ctx the geometric context
/// @param det the detector
/// @param ptc the truth particle
/// @param hits all hits in the event
/// @param n_hits_for_particle expected number of hits for the particle
///
/// @returns a vector of intersection candidate records
template <typename detector_t>
auto transcribe_to_trace(const typename detector_t::geometry_context ctx,
                         const detector_t& det,
                         const traccc::io::csv::particle& ptc,
                         const std::vector<traccc::io::csv::hit>& hits,
                         const std::size_t n_hits_for_particle = 10u) {
    using intersection_t =
        typename candidate_type<detector_t>::intersection_type;

    detray::dvector<candidate_type<detector_t>> candidates{};
    candidates.reserve(n_hits_for_particle);

    // Fill the hits into the candidate trace
    for (const auto& h : hits) {
        if (h.particle_id != ptc.particle_id) {
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
        intersection_t intr{sf_desc,
                            path,
                            static_cast<nav_link_t>(bcd.volume()),
                            detray::intersection::status::e_inside,
                            true,
                            loc_pos};

        candidates.emplace_back(pos, dir, intr, ptc.q, vector::norm(mom));
    }

    // Sort records by intersection distance to origin of the trajectory
    auto sort_by_path = [&](const candidate_type<detector_t>& a,
                            const candidate_type<detector_t>& b) -> bool {
        return (a.intersection < b.intersection);
    };

    std::ranges::stable_sort(candidates, sort_by_path);

    return candidates;
}

/// Transcribe the measurements for a particle to a candidate trace for the
/// detray propagation validation tools
///
/// @param ctx the geometric context
/// @param det the detector
/// @param ptc the truth particle
/// @param ptc_to_meas_map map the truth particle to its measurements
/// @param n_meas_for_particle expected number of ,easurements for the particle
///
/// @returns a vector of intersection candidate records
template <typename detector_t>
auto transcribe_to_trace(
    const typename detector_t::geometry_context ctx, const detector_t& det,
    const traccc::particle& ptc,
    const std::map<traccc::particle, std::vector<traccc::measurement>>&
        ptc_to_meas_map,
    const std::size_t n_meas_for_particle = 10u) {

    using intersection_t =
        typename candidate_type<detector_t>::intersection_type;

    vecmem::vector<candidate_type<detector_t>> candidates{};
    candidates.reserve(n_meas_for_particle);

    // TODO: Not accurate for every measurement
    const scalar p{vector::norm(ptc.momentum)};
    const scalar q{ptc.charge};

    // Fill the hits into the candidate trace
    for (const auto& meas : ptc_to_meas_map.at(ptc)) {
        // Corresponding surface
        const detray::geometry::barcode bcd{meas.surface_link};
        const auto sf_desc = det.surfaces().at(bcd.index());
        const auto sf = detray::tracking_surface{det, bcd};

        /*scalar local_1{meas.local[1]};
        using annulus_t =
            detray::mask<detray::annulus2D, traccc::default_algebra>;
        if (sf_desc.mask().id() ==
            detector_t::masks::template get_id<annulus_t>()) {
            std::cout << "Before: " << local_1 << std::endl;
            local_1 *= traccc::unit<scalar>::degree;
            std::cout << "After: " << local_1 << std::endl;
            std::cout << std::boolalpha << "Inside " <<
        sf.is_inside(point3{meas.local[0], local_1, 0.f}, 0.f) << std::endl;
        }
        point2 loc{meas.local[0], local_1};*/
        // TODO: Use correct track direction at measurement for line sf.
        const vector3 dir{vector::normalize(ptc.momentum)};
        const point3 glob_pos{sf.local_to_global(ctx, meas.local, dir)};
        // const point3 glob_pos{sf.local_to_global(ctx, loc, dir)};

        // Rough estimate of intersection distance from origin
        const scalar path{vector::norm(glob_pos)};

        // Build an intersection
        using nav_link_t = typename intersection_t::nav_link_t;
        intersection_t intr{sf_desc,
                            {path, {meas.local[0], meas.local[1], 0.f}},
                            static_cast<nav_link_t>(bcd.volume()),
                            detray::intersection::status::e_inside,
                            true};

        // TODO: Don't use initial particle momentum
        candidates.emplace_back(glob_pos, dir, intr, q, p);
    }

    // Sort records by intersection distance to origin of the trajectory
    auto sort_by_path = [&](const candidate_type<detector_t>& a,
                            const candidate_type<detector_t>& b) -> bool {
        return (a.intersection < b.intersection);
    };

    std::ranges::stable_sort(candidates, sort_by_path);

    return candidates;
}

}  // namespace traccc::propagation_validator
