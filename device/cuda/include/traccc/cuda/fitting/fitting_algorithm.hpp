/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

// traccc library include(s).
#include "traccc/utils/memory_resource.hpp"

namespace traccc::cuda {

/// Fitting algorithm for a set of tracks
template <typename fitter_t>
class fitting_algorithm
    : public algorithm<track_state_container_types::buffer(
          const typename fitter_t::detector_type::view_type&,
          const typename fitter_t::bfield_type&,
          const typename track_candidate_container_types::const_view&)>,
      public messaging {

    public:
    using algebra_type = typename fitter_t::algebra_type;
    /// Configuration type
    using config_type = typename fitter_t::config_type;

    /// Constructor for the fitting algorithm
    ///
    /// @param cfg  Configuration object
    /// @param mr   The memory resource to use
    /// @param copy Copy object
    /// @param str  Cuda stream object
    fitting_algorithm(
        const config_type& cfg, const traccc::memory_resource& mr,
        vecmem::copy& copy, stream& str,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Run the algorithm
    track_state_container_types::buffer operator()(
        const typename fitter_t::detector_type::view_type& det_view,
        const typename fitter_t::bfield_type& field_view,
        const typename track_candidate_container_types::const_view&
            track_candidates_view) const override;

    private:
    /// Config object
    config_type m_cfg;
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
