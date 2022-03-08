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

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

/// track parameter estimation for sycl
struct track_params_estimation
    : public algorithm<host_bound_track_parameters_collection(
          host_spacepoint_container&&, host_seed_collection&&)> {
    public:
    /// Constructor for track_params_estimation
    ///
    /// @param mr is the memory resource
    /// @param q sycl queue for kernel scheduling
    track_params_estimation(vecmem::memory_resource& mr, queue_wrapper queue);

    /// Callable operator for track_params_esitmation
    ///
    /// @param input_type is the seed container
    ///
    /// @return vector of bound track parameters
    output_type operator()(host_spacepoint_container&& spacepoints,
                           host_seed_collection&& seeds) const override;

    private:
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl
