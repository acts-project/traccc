/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

namespace traccc::device {

/// Algorithm to copy a container from the host to the device
///
/// Specialisations of this algorithm can be used to copy a container to a
/// device for a subsequent algorithm that expects its input to already be on
/// the device that it wants to run on.
///
/// Note that the input type is not a "host type", but rather a constant view.
/// This is meant to allow using this algorithm on top of more complicated
/// data objects.
///
/// @tparam CONTAINER_TYPES One of the "container types" traits
///
template <typename CONTAINER_TYPES>
class container_h2d_copy_alg
    : public algorithm<typename CONTAINER_TYPES::buffer(
          const typename CONTAINER_TYPES::const_view&)> {

    public:
    /// Helper type declaration for the input type
    typedef const typename CONTAINER_TYPES::const_view& input_type;
    /// Help the compiler understand what @c output_type is
    using output_type = typename algorithm<typename CONTAINER_TYPES::buffer(
        const typename CONTAINER_TYPES::const_view&)>::output_type;

    /// Constructor with the needed resources
    container_h2d_copy_alg(const memory_resource& mr, vecmem::copy& copy);

    /// Function executing the copy to the device
    virtual output_type operator()(input_type input) const override;

    private:
    /// The memory resource(s) to use
    memory_resource m_mr;
    /// The copy object to use
    vecmem::copy& m_copy;

};  // class container_h2d_copy_alg

}  // namespace traccc::device

// Include the implementation.
#include "traccc/device/impl/container_h2d_copy_alg.ipp"
