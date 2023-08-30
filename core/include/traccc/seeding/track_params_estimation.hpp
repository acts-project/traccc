/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/cell.hpp"
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
          const seed_collection_types::host&,
          const cell_module_collection_types::host&, const vector3&,
          const std::array<traccc::scalar, traccc::e_bound_size>&)> {

    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    track_params_estimation(vecmem::memory_resource& mr);

    /// Callable operator for track_params_esitmation
    ///
    /// @param spacepoints All spacepoints of the event
    /// @param seeds The reconstructed track seeds of the event
    /// @param modules Geometry module vector
    /// @param bfield (Temporary) Magnetic field vector
    /// @param stddev standard deviation for setting the covariance (Default
    /// value from arXiv:2112.09470v1)
    /// @return A vector of bound track parameters
    ///
    output_type operator()(
        const spacepoint_collection_types::host& spacepoints,
        const seed_collection_types::host& seeds,
        const cell_module_collection_types::host& modules,
        const vector3& bfield,
        const std::array<traccc::scalar, traccc::e_bound_size>& stddev = {
            0.02 * detray::unit<traccc::scalar>::mm,
            0.03 * detray::unit<traccc::scalar>::mm,
            1. * detray::unit<traccc::scalar>::degree,
            1. * detray::unit<traccc::scalar>::degree,
            0.01 / detray::unit<traccc::scalar>::GeV,
            1 * detray::unit<traccc::scalar>::ns}) const override;

    private:
    /// The memory resource to use in the algorithm
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class track_params_estimation

}  // namespace traccc
