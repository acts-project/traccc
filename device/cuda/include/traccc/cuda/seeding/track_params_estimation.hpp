/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/seeding/detail/track_params_estimation_config.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

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
          const measurement_collection_types::const_view&,
          const edm::spacepoint_collection::const_view&,
          const edm::seed_collection::const_view&, const vector3&)>,
      public messaging {

    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    track_params_estimation(
        const track_params_estimation_config& config,
        const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Callable operator for track_params_estimation
    ///
    /// @param measurements All measurements of the event
    /// @param spacepoints All spacepoints of the event
    /// @param seeds The reconstructed track seeds of the event
    /// @param bfield (Temporary) Magnetic field vector
    /// @param stddev standard deviation for setting the covariance (Default
    /// value from arXiv:2112.09470v1)
    /// @return A vector of bound track parameters
    ///
    output_type operator()(
        const measurement_collection_types::const_view& measurements,
        const edm::spacepoint_collection::const_view& spacepoints,
        const edm::seed_collection::const_view& seeds,
        const vector3& bfield) const override;

    private:
    const track_params_estimation_config m_config;

    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// The copy object to use
    vecmem::copy& m_copy;
    /// The CUDA stream to use
    stream& m_stream;

    /// Warp size of the GPU being used
    unsigned int m_warp_size;
};

}  // namespace traccc::cuda
