/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

#include "traccc/alpaka/utils/definitions.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::alpaka {

/// track parameter estimation for alpaka
struct track_params_estimation
    : public algorithm<bound_track_parameters_collection_types::buffer(
          const spacepoint_collection_types::const_view&,
          const seed_collection_types::const_view&, const vector3&)> {

    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    track_params_estimation(const traccc::memory_resource& mr, vecmem::copy& copy);

    /// Callable operator for track_params_estimation
    ///
    /// @param spacepoints_view   is the view of the spacepoint collection
    /// @param seeds_view        is the view of the seed collection
    /// @param bfield            the current b-field
    /// @return                  vector of bound track parameters
    ///
    output_type operator()(
        const spacepoint_collection_types::const_view& spacepoints_view,
        const seed_collection_types::const_view& seeds_view,
        const vector3& bfield) const override;

    private:
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// Copy object used by the algorithm
    vecmem::copy& m_copy;
};

}  // namespace traccc::alpaka
