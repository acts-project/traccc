/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/sycl/utils/queue_wrapper.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

/// Algorithm forming space points out of measurements
///
/// This algorithm performs the local-to-global transformation of the 2D
/// measurements made on every detector module, into 3D spacepoint coordinates.
///
template <typename detector_t>
class spacepoint_formation_algorithm
    : public algorithm<spacepoint_collection_types::buffer(
          const typename detector_t::view_type&,
          const measurement_collection_types::const_view&)> {

    public:
    /// Constructor for spacepoint_formation
    ///
    /// @param mr    the memory resource
    /// @param copy  vecmem copy object
    /// @param queue is a wrapper for the sycl queue for kernel
    ///
    spacepoint_formation_algorithm(const traccc::memory_resource& mr,
                                   vecmem::copy& copy, queue_wrapper queue);

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
    std::reference_wrapper<vecmem::copy> m_copy;
    /// SYCL queue object
    mutable queue_wrapper m_queue;
};

}  // namespace traccc::sycl