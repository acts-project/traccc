/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::cuda::experimental {

/// Algorithm forming space points out of measurements
///
/// This algorithm performs the local-to-global transformation of the 2D
/// measurements made on every detector module, into 3D spacepoint coordinates.
///
template <typename detector_t>
class spacepoint_formation
    : public algorithm<spacepoint_collection_types::buffer(
          const typename detector_t::view_type&,
          const measurement_collection_types::const_view&)> {

    public:
    /// Constructor for spacepoint_formation
    ///
    /// @param mr   the memory resource
    /// @param copy vecmem copy object
    /// @param str  cuda stream
    ///
    spacepoint_formation(const traccc::memory_resource& mr, vecmem::copy& copy,
                         stream& str);

    /// Callable operator for spacepoint formation
    ///
    /// @param det_view     a detector view object
    /// @param measurements  a collection of measurements
    /// @return a spacepoint collection (buffer)
    spacepoint_collection_types::buffer operator()(
        const typename detector_t::view_type& det_view,
        const measurement_collection_types::const_view& measurements_view)
        const override;

    private:
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// The copy object to use
    vecmem::copy& m_copy;
    /// The CUDA stream to use
    stream& m_stream;
};

}  // namespace traccc::cuda::experimental