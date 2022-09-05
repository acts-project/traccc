/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

/// track parameter estimation for sycl
struct track_params_estimation
    : public algorithm<host_bound_track_parameters_collection(
          const spacepoint_container_types::const_view&,
          const vecmem::data::vector_view<const seed>&)>,
      public algorithm<host_bound_track_parameters_collection(
          const spacepoint_container_types::buffer&,
          const vecmem::data::vector_buffer<seed>&)> {

    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr       is a struct of memory resources (shared or
    /// host & device)
    /// @param queue    is a wrapper for the sycl queue for kernel
    /// invocation
    track_params_estimation(const traccc::memory_resource& mr,
                            queue_wrapper queue);

    /// Callable operator for track_params_esitmation
    ///
    /// @param spaepoints_view   is the view of the spacepoint container
    /// @param seeds_view        is the view of the seed container
    /// @return                  vector of bound track parameters
    ///
    host_bound_track_parameters_collection operator()(
        const spacepoint_container_types::const_view& spacepoints_view,
        const vecmem::data::vector_view<const seed>& seeds_view) const override;

    /// Callable operator for track_params_esitmation
    ///
    /// @param spaepoints_buffer   is the buffer of the spacepoint container
    /// @param seeds_buffer        is the buffer of the seed container
    /// @return                    vector of bound track parameters
    ///
    host_bound_track_parameters_collection operator()(
        const spacepoint_container_types::buffer& spacepoints_buffer,
        const vecmem::data::vector_buffer<seed>& seeds_buffer) const override;

    private:
    /// Implementation for the public track params estimation operators
    host_bound_track_parameters_collection operator()(
        const spacepoint_container_types::const_view& spacepoints_view,
        const vecmem::data::vector_view<const seed>& seeds_view,
        std::size_t seeds_size) const;

    // Private member variables
    traccc::memory_resource m_mr;
    mutable queue_wrapper m_queue;
    std::unique_ptr<vecmem::copy> m_copy;
};

}  // namespace traccc::sycl
