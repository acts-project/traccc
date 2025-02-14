/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/cuda/utils/stream.hpp"

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::cuda {

/// Algorithm forming space points out of measurements
///
/// This algorithm performs the local-to-global transformation of the 2D
/// measurements made on every detector module, into 3D spacepoint coordinates.
///
template <typename detector_t>
class spacepoint_formation_algorithm
    : public algorithm<edm::spacepoint_collection::buffer(
          const typename detector_t::view_type&,
          const measurement_collection_types::const_view&)>,
      public messaging {

    public:
    /// Constructor for spacepoint_formation
    ///
    /// @param mr is the memory resource
    ///
    spacepoint_formation_algorithm(
        const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Callable operator for the space point formation, based on one single
    /// module
    ///
    /// @param det_view     A view type object of tracking geometry
    /// @param measurements All reconstructed measurements in an event
    /// @return A spacepoint container, with one spacepoint for every
    ///         measurement
    ///
    edm::spacepoint_collection::buffer operator()(
        const typename detector_t::view_type& det_view,
        const measurement_collection_types::const_view& measurements)
        const override;

    private:
    /// The memory resource(s) to use
    traccc::memory_resource m_mr;
    /// The copy object to use
    std::reference_wrapper<vecmem::copy> m_copy;
    /// The CUDA stream to use
    std::reference_wrapper<stream> m_stream;
};  // class spacepoint_formation_algorithm

}  // namespace traccc::cuda
