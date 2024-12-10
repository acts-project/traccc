/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cassert>
#include <type_traits>

#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

namespace vecmem::details {
/**
 * @brief A deleter class for trivial allocations.
 *
 * This class provides all the necessary functionality to allow unique pointers
 * to deallocate allocations of trivial types, as well as arrays of them.
 *
 * @tparam T The type to deallocate.
 *
 * @note This deleter class supports non-array types, array types of unknown
 * bound, but not array types of known bound.
 *
 * @note This deleter class only supports types that are both trivially
 * constructible and trivially destructible.
 */
template <typename T>
struct unique_alloc_deleter {
    /*
     * Similarly, we cannot destroy objects, so we need to ensure that
     * deallocation of objects is semantically equivalent to their destruction.
     */
    static_assert(std::is_trivially_destructible_v<std::remove_extent_t<T>>,
                  "Allocation pointer type must be trivially destructible.");

public:
    /**
     * @brief Default-construct a new unique allocation deleter.
     *
     * This constructor should only really be used in cases where we are
     * nulling a unique pointer.
     */
    unique_alloc_deleter(void) = default;

    /**
     * @brief Construct a new unique allocation deleter.
     *
     * This is the standard constructor for this class, which stores all the
     * necessary data to deallocate an allocation.
     *
     * @param mr The memory resource to use for deallocation.
     * @param p The pointer to deallocate.
     * @param s The size of the allocation.
     * @param a The alignment of the allocation.
     */
    unique_alloc_deleter(memory_resource& mr, std::size_t s, std::size_t a = 0)
        : m_mr(&mr), m_size(s), m_align(a) {}

    /**
     * @brief Copy a unique allocation deleter.
     *
     * @param i The allocation deleter to copy.
     */
    unique_alloc_deleter(const unique_alloc_deleter& i) = default;

    /**
     * @brief Move a unique allocation deleter.
     *
     * @param i The allocation deleter to move.
     */
    unique_alloc_deleter(unique_alloc_deleter&& i) = default;

    /**
     * @brief Copy-assign a unique allocation deleter.
     *
     * @param i The object to copy into the current one.
     *
     * @return A reference to this object.
     */
    unique_alloc_deleter& operator=(const unique_alloc_deleter& i) = default;

    /**
     * @brief Move-assign a unique allocation deleter.
     *
     * @param i The object to move into the current one.
     *
     * @return A reference to this object.
     */
    unique_alloc_deleter& operator=(unique_alloc_deleter&& i) = default;

    /**
     * @brief Activate the deletion mechanism of the deleter.
     *
     * @param p The pointer to deallocate.
     */
    void operator()(void* p) const {
        assert(m_mr != nullptr);

        /*
         * As before, if this happens... Something has gone VERY wrong.
         */
        if (m_mr == nullptr) {
            return;
        }

        /*
         * Deallocate the memory that we were using.
         */
        if (m_align > 0) {
            m_mr->deallocate(p, m_size, m_align);
        } else {
            m_mr->deallocate(p, m_size);
        }
    }

private:
    memory_resource* m_mr;
    std::size_t m_size;
    std::size_t m_align;
};
}  // namespace vecmem::details
