/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <cstddef>

namespace vecmem {

/**
 * @brief This memory resource forwards allocation and deallocation requests to
 * the upstream resource.
 *
 * This allocator is here to act as the unit in the monoid of memory resources.
 * It serves only a niche practical purpose.
 */
class identity_memory_resource final : public details::memory_resource_base {

public:
    /**
     * @brief Constructs the identity memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    VECMEM_CORE_EXPORT
    identity_memory_resource(memory_resource& upstream);
    /// Destructor
    VECMEM_CORE_EXPORT
    ~identity_memory_resource();

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate memory with the upstream resource
    VECMEM_CORE_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// De-allocate a previously allocated memory block
    VECMEM_CORE_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;
    /// Compare the equality of @c *this memory resource with another
    VECMEM_CORE_EXPORT
    virtual bool do_is_equal(
        const memory_resource&) const noexcept override final;

    /// @}

    /// The upstream memory resource to use
    memory_resource& m_upstream;

};  // class identity_memory_resource

}  // namespace vecmem
