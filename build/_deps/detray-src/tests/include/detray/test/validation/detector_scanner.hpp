/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/geometry/tracking_surface.hpp"
#include "detray/navigation/detail/trajectories.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/navigation/intersection_kernel.hpp"
#include "detray/navigation/intersector.hpp"
#include "detray/tracks/free_track_parameters.hpp"

// Detray IO include(s)
#include "detray/io/csv/intersection2D.hpp"
#include "detray/io/csv/track_parameters.hpp"

// System include(s)
#include <algorithm>
#include <type_traits>

namespace detray {

/// Record of a surface intersection along a track
template <typename detector_t>
struct intersection_record {
    using algebra_t = typename detector_t::algebra_type;
    using scalar_t = dscalar<algebra_t>;
    using track_parameter_type = free_track_parameters<algebra_t>;
    using intersection_type =
        intersection2D<typename detector_t::surface_type, algebra_t, true>;

    /// The charge associated with the track parameters
    scalar_t charge;
    /// Current global track parameters
    track_parameter_type track_param;
    /// Index of the volume the intersection was found in
    dindex vol_idx;
    /// The intersection result, including the surface descriptor
    intersection_type intersection;
};

/// @brief struct that holds functionality to shoot a parametrized particle
/// trajectory through a detector.
///
/// Records intersections with every detector surface along the trajectory.
template <typename trajectory_t>
struct brute_force_scan {

    template <typename D>
    using intersection_trace_type = std::vector<intersection_record<D>>;
    using trajectory_type = trajectory_t;

    template <typename detector_t>
    inline auto operator()(const typename detector_t::geometry_context ctx,
                           const detector_t &detector, const trajectory_t &traj,
                           const std::array<typename detector_t::scalar_type, 2>
                               mask_tolerance = {0.f, 0.f},
                           const typename detector_t::scalar_type p =
                               1.f *
                               unit<typename detector_t::scalar_type>::GeV) {

        using algebra_t = typename detector_t::scalar_type;
        using scalar_t = dscalar<algebra_t>;
        using sf_desc_t = typename detector_t::surface_type;
        using nav_link_t = typename detector_t::surface_type::navigation_link;

        using intersection_t =
            intersection2D<sf_desc_t, typename detector_t::algebra_type, true>;

        using intersection_kernel_t = intersection_initialize<intersector>;

        intersection_trace_type<detector_t> intersection_trace;

        const auto &trf_store = detector.transform_store();

        assert(p > 0.f);
        const scalar_t q{p * traj.qop()};

        std::vector<intersection_t> intersections{};
        intersections.reserve(100u);

        // Loop over all surfaces in the detector
        for (const sf_desc_t &sf_desc : detector.surfaces()) {
            // Retrieve candidate(s) from the surface
            const auto sf = tracking_surface{detector, sf_desc};
            sf.template visit_mask<intersection_kernel_t>(
                intersections, traj, sf_desc, trf_store, ctx,
                sf.is_portal() ? std::array<scalar_t, 2>{0.f, 0.f}
                               : mask_tolerance);

            // Candidate is invalid if it lies in the opposite direction
            for (auto &sfi : intersections) {
                if (sfi.direction) {
                    sfi.sf_desc = sf_desc;
                    // Record the intersection
                    intersection_trace.push_back(
                        {q,
                         {traj.pos(sfi.path), 0.f, p * traj.dir(sfi.path), q},
                         sf.volume(),
                         sfi});
                }
            }
            intersections.clear();
        }

        // Save initial track position as dummy intersection record
        const auto &first_record = intersection_trace.front();
        intersection_t start_intersection{};
        start_intersection.sf_desc = first_record.intersection.sf_desc;
        start_intersection.sf_desc.set_id(surface_id::e_passive);
        start_intersection.sf_desc.set_index(dindex_invalid);
        start_intersection.sf_desc.material().set_id(
            detector_t::materials::id::e_none);
        start_intersection.path = 0.f;
        start_intersection.local = {0.f, 0.f, 0.f};
        start_intersection.volume_link =
            static_cast<nav_link_t>(first_record.vol_idx);

        intersection_trace.insert(intersection_trace.begin(),
                                  intersection_record<detector_t>{
                                      q,
                                      {traj.pos(), 0.f, p * traj.dir(), q},
                                      first_record.vol_idx,
                                      start_intersection});

        return intersection_trace;
    }
};

template <typename algebra_t>
using ray_scan = brute_force_scan<detail::ray<algebra_t>>;

template <typename algebra_t>
using helix_scan = brute_force_scan<detail::helix<algebra_t>>;

/// Run a scan on detector object by shooting test particles through it
namespace detector_scanner {

template <template <typename> class scan_type, typename detector_t,
          typename trajectory_t, typename... Args>
inline auto run(const typename detector_t::geometry_context gctx,
                const detector_t &detector, const trajectory_t &traj,
                Args &&... args) {

    using algebra_t = typename detector_t::algebra_type;

    auto intersection_record = scan_type<algebra_t>{}(
        gctx, detector, traj, std::forward<Args>(args)...);

    using record_t = typename decltype(intersection_record)::value_type;

    // Sort intersections by distance to origin of the trajectory
    auto sort_path = [&](const record_t &a, const record_t &b) -> bool {
        return (a.intersection < b.intersection);
    };
    std::ranges::stable_sort(intersection_record, sort_path);

    // Make sure the intersection record terminates at world portals
    auto is_world_exit = [](const record_t &r) {
        return r.intersection.volume_link ==
               detray::detail::invalid_value<decltype(
                   r.intersection.volume_link)>();
    };

    if (auto it = std::ranges::find_if(intersection_record, is_world_exit);
        it != intersection_record.end()) {
        auto n{static_cast<std::size_t>(it - intersection_record.begin())};
        intersection_record.resize(n + 1u);
    }

    return intersection_record;
}

/// Write the @param intersection_traces to file
template <typename detector_t>
inline auto write_intersections(
    const std::string &intersection_file_name,
    const std::vector<std::vector<intersection_record<detector_t>>>
        &intersection_traces) {

    using record_t = intersection_record<detector_t>;
    using intersection_t = typename record_t::intersection_type;

    std::vector<std::vector<intersection_t>> intersections{};

    // Split data
    for (const auto &trace : intersection_traces) {

        intersections.push_back({});
        intersections.back().reserve(trace.size());

        for (const auto &record : trace) {
            intersections.back().push_back(record.intersection);
        }
    }

    // Write to file
    io::csv::write_intersection2D(intersection_file_name, intersections);
}

/// Write the @param intersection_traces to file
template <typename detector_t>
inline auto write_tracks(
    const std::string &track_param_file_name,
    const std::vector<std::vector<intersection_record<detector_t>>>
        &intersection_traces) {

    using scalar_t = typename detector_t::scalar_type;
    using record_t = intersection_record<detector_t>;
    using track_param_t = typename record_t::track_parameter_type;

    std::vector<std::vector<std::pair<scalar_t, track_param_t>>> track_params{};

    // Split data
    for (const auto &trace : intersection_traces) {
        track_params.push_back({});
        track_params.back().reserve(trace.size());

        for (const auto &record : trace) {
            track_params.back().emplace_back(record.charge, record.track_param);
        }
    }

    // Write to file
    io::csv::write_free_track_params(track_param_file_name, track_params);
}

/// Read the @param intersection_record from file
template <typename detector_t>
inline auto read(const std::string &intersection_file_name,
                 const std::string &track_param_file_name,
                 std::vector<std::vector<intersection_record<detector_t>>>
                     &intersection_traces) {

    // Read from file
    auto intersections_per_track =
        io::csv::read_intersection2D<detector_t>(intersection_file_name);
    auto track_params_per_track =
        io::csv::read_free_track_params<detector_t>(track_param_file_name);

    if (intersections_per_track.size() != track_params_per_track.size()) {
        throw std::invalid_argument(
            "Detector scanner: intersection and track parameters collections "
            "have different size");
    }

    // Interleave data
    for (dindex trk_idx = 0u; trk_idx < intersections_per_track.size();
         ++trk_idx) {
        const auto &intersections = intersections_per_track[trk_idx];
        const auto &track_params = track_params_per_track[trk_idx];

        // Check track id
        if (intersections.size() != track_params.size()) {
            throw std::invalid_argument(
                "Detector scanner: Found different number of intersections and "
                "track parameters for track no." +
                std::to_string(trk_idx));
        }

        // Check for empty input traces
        if (intersections.empty()) {
            throw std::invalid_argument(
                "Detector scanner: Found empty trace no." +
                std::to_string(trk_idx));
        }

        // Add new trace
        if (intersection_traces.size() <= trk_idx) {
            intersection_traces.push_back({});
        }

        // Add records to trace
        for (dindex i = 0u; i < intersections.size(); ++i) {

            intersection_traces[trk_idx].push_back(
                intersection_record<detector_t>{
                    track_params[i].first, track_params[i].second,
                    intersections[i].sf_desc.volume(), intersections[i]});
        }
    }
}

}  // namespace detector_scanner

}  // namespace detray
