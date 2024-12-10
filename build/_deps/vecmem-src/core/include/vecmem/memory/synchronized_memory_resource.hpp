/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <functional>
#include <mutex>

namespace vecmem {

/// A memory resource that synchronizes the operations of an upstream resource
///
/// This memory resource is a wrapper around another memory resource that
/// synchronizes all operations of the upstream resource. This is useful for
/// memory resources that are themselves not thread-safe, but need to be used in
/// a multi-threaded environment.
///
class synchronized_memory_resource final : public memory_resource {

public:
    /// Constructor around an upstream memory resource
    VECMEM_CORE_EXPORT
    synchronized_memory_resource(memory_resource& upstream);
    /// Move constructor
    VECMEM_CORE_EXPORT
    synchronized_memory_resource(synchronized_memory_resource&& parent);
    /// Copy constructor
    VECMEM_CORE_EXPORT
    synchronized_memory_resource(const synchronized_memory_resource& parent);

    /// Destructor
    VECMEM_CORE_EXPORT
    virtual ~synchronized_memory_resource();

    /// Move assignment operator
    VECMEM_CORE_EXPORT
    synchronized_memory_resource& operator=(synchronized_memory_resource&& rhs);
    /// Copy assignment operator
    VECMEM_CORE_EXPORT
    synchronized_memory_resource& operator=(
        const synchronized_memory_resource& rhs);

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate memory with one of the underlying resources
    VECMEM_CORE_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// De-allocate a previously allocated memory block
    VECMEM_CORE_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;
    /// Compare the equality of @c *this memory resource with another
    VECMEM_CORE_EXPORT
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

    /// The upstream memory resource
    std::reference_wrapper<memory_resource> m_upstream;
    /// The mutex to synchronize the upstream memory resource's operations
    std::mutex m_mutex;

};  // class synchronized_memory_resource

}  // namespace vecmem
