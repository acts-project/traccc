/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/utils/algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc {

/// Track parameter estimation algorithm
///
/// Transcribed from Acts/Seeding/EstimateTrackParamsFromSeed.hpp.
///
class track_params_estimation
    : public algorithm<bound_track_parameters_collection_types::host(
          const spacepoint_collection_types::host&,
          const seed_collection_types::host&)> {

    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    track_params_estimation(vecmem::memory_resource& mr);

    /// Callable operator for track_params_esitmation
    ///
    /// @param spacepoints All spacepoints of the event
    /// @param seeds The reconstructed track seeds of the event
    /// @return A vector of bound track parameters
    ///
    output_type operator()(
        const spacepoint_collection_types::host& spacepoints,
        const seed_collection_types::host& seeds) const override;

    private:
    /// The memory resource to use in the algorithm
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class track_params_estimation

}  // namespace traccc
