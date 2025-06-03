/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate_collection.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::edm {

/// Convenience type for describing all components of a track candidate
/// collection
template <typename ALGEBRA>
struct track_candidate_container {

    struct view {
        /// The track candidates
        track_candidate_collection<ALGEBRA>::view tracks;
        /// Measurements referenced by the tracks
        measurement_collection_types::const_view measurements;
    };

    struct const_view {
        /// The track candidates
        track_candidate_collection<ALGEBRA>::const_view tracks;
        /// Measurements referenced by the tracks
        measurement_collection_types::const_view measurements;
    };

    struct host {
        /// Constructor
        host(vecmem::memory_resource& mr) : tracks{mr}, measurements{&mr} {}
        /// Move constructor
        host(track_candidate_collection<ALGEBRA>::host&& other_tracks,
             measurement_collection_types::host&& other_measurements)
            : tracks{std::move(other_tracks)},
              measurements{std::move(other_measurements)} {}

        /// The track candidates
        track_candidate_collection<ALGEBRA>::host tracks;
        /// Measurements referenced by the tracks
        measurement_collection_types::host measurements;
    };

    struct buffer {
        /// The track candidates
        track_candidate_collection<ALGEBRA>::buffer tracks;
        /// Measurements referenced by the tracks
        measurement_collection_types::buffer measurements;
    };

    struct data {
        /// The track candidates
        track_candidate_collection<ALGEBRA>::data tracks;
        /// Measurements referenced by the tracks
        measurement_collection_types::view measurements;
    };

    struct const_data {
        /// The track candidates
        track_candidate_collection<ALGEBRA>::const_data tracks;
        /// Measurements referenced by the tracks
        measurement_collection_types::const_view measurements;
    };

};  // struct track_candidate_container

}  // namespace traccc::edm
