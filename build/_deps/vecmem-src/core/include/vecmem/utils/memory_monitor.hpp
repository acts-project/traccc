/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/instrumenting_memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <cstddef>

namespace vecmem {

/// Class collecting some basic set of memory allocation statistics
///
/// Objects of this class can be used together with
/// @c vecmem::instrumenting_memory_resource to easily access a common set of
/// useful performance metrics about an application.
///
/// Note that the lifetime of this object must be at least as long as the
/// lifetime of the connected memory resource!
///
class VECMEM_CORE_EXPORT memory_monitor {

public:
    /// Constructor with a memory resource reference
    memory_monitor(instrumenting_memory_resource& resource);

    /// Get the total amount of allocations
    std::size_t total_allocation() const;
    /// Get the outstanding allocation left after all operations
    std::size_t outstanding_allocation() const;
    /// Get the average allocation size
    std::size_t average_allocation() const;
    /// Get the maximal concurrent allocation
    std::size_t maximal_allocation() const;

private:
    /// @name Function(s) implementing the "monitor interface"
    /// @{

    /// Function called after successful memory allocations
    void post_allocate(std::size_t size, std::size_t align, void* ptr);
    /// Function called before memory de-allocations
    void pre_deallocate(void* ptr, std::size_t size, std::size_t align);

    /// @}

    /// The number of allocations
    std::size_t m_n_alloc = 0;
    /// Total allocation
    std::size_t m_total_alloc = 0;
    /// Outstanding allocation
    std::size_t m_outstanding_alloc = 0;
    /// Maximum allocation
    std::size_t m_maximum_alloc = 0;

};  // class memory_monitor

}  // namespace vecmem
