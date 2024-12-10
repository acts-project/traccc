/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <cstddef>

namespace vecmem::details {

/// Base class for implementations of the @c vecmem::memory_resource interface
///
/// It's mainly just a convenience class for providing a common implementation
/// of the @c vecmem::memory_resource::is_equal(...) function for the derived
/// types.
///
class memory_resource_base : public memory_resource {

protected:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Compare the equality of @c *this memory resource with another
    ///
    /// @param other The other memory resource to compare with
    /// @returns @c true if the two memory resources are equal, @c false
    ///          otherwise
    ///
    VECMEM_CORE_EXPORT
    virtual bool do_is_equal(const memory_resource& other) const noexcept;

    /// @}

};  // class memory_resource_base

}  // namespace vecmem::details
