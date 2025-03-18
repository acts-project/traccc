/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include <vecmem/utils/copy.hpp>

#include "traccc/edm/measurement.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

namespace traccc::alpaka {

/// track parameter estimation for alpaka
struct track_params_estimation
    : public algorithm<bound_track_parameters_collection_types::buffer(
          const measurement_collection_types::const_view&,
          const edm::spacepoint_collection::const_view&,
          const edm::seed_collection::const_view&, const vector3&,
          const std::array<traccc::scalar, traccc::e_bound_size>&)>,
      public messaging {

    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    track_params_estimation(
        const traccc::memory_resource& mr, vecmem::copy& copy,
        std::unique_ptr<const Logger> ilogger = getDummyLogger().clone());

    /// Callable operator for track_params_estimation
    ///
    /// @param measurements All measurements of the event
    /// @param spacepoints All spacepoints of the event
    /// @param seeds The reconstructed track seeds of the event
    /// @param modules Geometry module vector
    /// @param bfield (Temporary) Magnetic field vector
    /// @param stddev standard deviation for setting the covariance (Default
    /// value from arXiv:2112.09470v1)
    /// @return A vector of bound track parameters
    ///
    output_type operator()(
        const measurement_collection_types::const_view& measurements,
        const edm::spacepoint_collection::const_view& spacepoints,
        const edm::seed_collection::const_view& seeds, const vector3& bfield,
        const std::array<traccc::scalar, traccc::e_bound_size>& = {
            0.02f * traccc::unit<traccc::scalar>::mm,
            0.03f * traccc::unit<traccc::scalar>::mm,
            1.f * traccc::unit<traccc::scalar>::degree,
            1.f * traccc::unit<traccc::scalar>::degree,
            0.01f / traccc::unit<traccc::scalar>::GeV,
            1.f * traccc::unit<traccc::scalar>::ns}) const override;

    private:
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// Copy object used by the algorithm
    ::vecmem::copy& m_copy;
};

}  // namespace traccc::alpaka
