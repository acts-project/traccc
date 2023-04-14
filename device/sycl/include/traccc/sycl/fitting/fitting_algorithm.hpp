/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/sycl/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

/// Fitting algorithm for a set of tracks
template <typename fitter_t>
class fitting_algorithm
    : public algorithm<track_state_container_types::buffer(
          const typename fitter_t::detector_type::detector_view_type&,
          const vecmem::data::jagged_vector_view<
              typename fitter_t::intersection_type>&,
          const typename track_candidate_container_types::const_view&)> {

    public:
    /// Constructor for the fitting algorithm
    ///
    /// @param mr The memory resource to use
    /// @param queue is a wrapper for the sycl queue for kernel invocation
    fitting_algorithm(const traccc::memory_resource& mr, queue_wrapper queue);

    /// Run the algorithm
    track_state_container_types::buffer operator()(
        const typename fitter_t::detector_type::detector_view_type& det_view,
        const vecmem::data::jagged_vector_view<
            typename fitter_t::intersection_type>& navigation_buffer,
        const typename track_candidate_container_types::const_view&
            track_candidates_view) const override;

    private:
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// Queue wrapper
    mutable queue_wrapper m_queue;
    /// Copy object used by the algorithm
    std::unique_ptr<vecmem::copy> m_copy;
};

}  // namespace traccc::sycl