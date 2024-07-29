/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::cuda {

/// track parameter estimation for cuda
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
struct track_params_estimation
    : public algorithm<bound_track_parameters_collection_types::buffer(
          const spacepoint_collection_types::const_view&,
          const seed_collection_types::const_view&, const vector3&,
          const std::array<traccc::scalar, traccc::e_bound_size>&)> {

    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    track_params_estimation(const traccc::memory_resource& mr,
                            vecmem::copy& copy, stream& str);

    /// Callable operator for track_params_estimation
    ///
    /// @param spacepoints All spacepoints of the event
    /// @param seeds The reconstructed track seeds of the event
    /// @param bfield (Temporary) Magnetic field vector
    /// @param stddev standard deviation for setting the covariance (Default
    /// value from arXiv:2112.09470v1)
    /// @return A vector of bound track parameters
    ///
    output_type operator()(
        const spacepoint_collection_types::const_view& spacepoints_view,
        const seed_collection_types::const_view& seeds_view,
        const vector3& bfield,
        const std::array<traccc::scalar, traccc::e_bound_size>& = {
            0.02f * detray::unit<traccc::scalar>::mm,
            0.03f * detray::unit<traccc::scalar>::mm,
            1.f * detray::unit<traccc::scalar>::degree,
            1.f * detray::unit<traccc::scalar>::degree,
            0.01f / detray::unit<traccc::scalar>::GeV,
            1.f * detray::unit<traccc::scalar>::ns}) const override;

    private:
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// The copy object to use
    vecmem::copy& m_copy;
    /// The CUDA stream to use
    stream& m_stream;

    /// Warp size of the GPU being used
    int m_warp_size;
};

}  // namespace traccc::cuda
