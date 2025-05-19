/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_fit_collection.hpp"
#include "traccc/edm/track_state_collection.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::edm {

/// Return type(s) for the track fitting algorithm(s)
template <typename ALGEBRA>
struct track_fit_container {

    struct view {
        /// The fitted tracks
        track_fit_collection<ALGEBRA>::view tracks;
        /// The track states used for the fit
        track_state_collection<ALGEBRA>::view states;
        /// The measurements used for the fit
        measurement_collection_types::const_view measurements;
    };

    struct const_view {
        /// The fitted tracks
        track_fit_collection<ALGEBRA>::const_view tracks;
        /// The track states used for the fit
        track_state_collection<ALGEBRA>::const_view states;
        /// The measurements used for the fit
        measurement_collection_types::const_view measurements;
    };

    struct host {
        /// Constructor using a memory resource
        explicit host(vecmem::memory_resource& mr) : tracks{mr}, states{mr} {}

        /// The fitted tracks
        track_fit_collection<ALGEBRA>::host tracks;
        /// The track states used for the fit
        track_state_collection<ALGEBRA>::host states;
    };

    struct buffer {
        /// The fitted tracks
        track_fit_collection<ALGEBRA>::buffer tracks;
        /// The track states used for the fit
        track_state_collection<ALGEBRA>::buffer states;
    };

    struct data {
        /// The fitted tracks
        track_fit_collection<ALGEBRA>::data tracks;
        /// The track states used for the fit
        track_state_collection<ALGEBRA>::data states;
    };

    struct const_data {
        /// The fitted tracks
        track_fit_collection<ALGEBRA>::const_data tracks;
        /// The track states used for the fit
        track_state_collection<ALGEBRA>::const_data states;
    };

};  // struct track_fit_container

}  // namespace traccc::edm
