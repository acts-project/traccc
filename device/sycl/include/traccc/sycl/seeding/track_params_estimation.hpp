/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::sycl {

/// track parameter estimation for sycl
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
    /// @param mr       is a struct of memory resources (shared or
    /// host & device)
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param queue    is a wrapper for the sycl queue for kernel
    /// invocation
    track_params_estimation(
        const traccc::memory_resource& mr, vecmem::copy& copy,
        queue_wrapper queue,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Callable operator for track_params_esitmation
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
        const edm::seed_collection::const_view& seeds, const vector3& bfield,
        const std::array<traccc::scalar, traccc::e_bound_size>& stddev = {
            0.02f * traccc::unit<traccc::scalar>::mm,
            0.03f * traccc::unit<traccc::scalar>::mm,
            1.f * traccc::unit<traccc::scalar>::degree,
            1.f * traccc::unit<traccc::scalar>::degree,
            0.01f / traccc::unit<traccc::scalar>::GeV,
            1.f * traccc::unit<traccc::scalar>::ns}) const override;

    private:
    // Private member variables
    traccc::memory_resource m_mr;
    mutable queue_wrapper m_queue;
    vecmem::copy& m_copy;
};

}  // namespace traccc::sycl
