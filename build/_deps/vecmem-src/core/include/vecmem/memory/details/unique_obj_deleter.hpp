/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"

// System include(s).
#include <cassert>
#include <type_traits>

namespace vecmem::details {
/**
 * @brief A deleter class for non-trivial objects.
 *
 * This class provides all the necessary functionality to allow unique pointers
 * to deallocate non-trivial objects, as well as arrays of them.
 *
 * Object pointers not only deallocate their memory, but they also cleanly
 * delete the object by callings its deconstructor.
 *
 * The design of this class is somewhat unconventional because it is stateful,
 * which most deleters are not. However, this is pretty much necessary we
 * cannot rely on the packed storage of allocation parameters that you see in
 * standard deleters. In particular, the pointer argument is not strictly
 * necessary, but it is included to make the class easier to debug.
 *
 * @tparam T The type to deallocate.
 *
 * @note This deleter class supports non-array types and array types of unknown
 * bound, but it does not support arrays of known bound.
 *
 * @warning This type should never be used with host-inaccessible memory! We
 * cannot check this at compile-time or at run-time, so it is the
 * responsibility of the user to ensure that this requirement is respected.
 */
template <typename T>
struct unique_obj_deleter {
public:
    static_assert(!std::is_array_v<T> ||
                      (std::is_array_v<T> && std::extent_v<T> == 0),
                  "Pointer type of unique object must be either non-array or "
                  "unbound array type.");

    using pointer_t =
        std::conditional_t<std::is_array_v<T>, std::decay_t<T>, T*>;
    using storage_t = std::remove_pointer_t<pointer_t>;

    /**
     * @brief Default-construct a new unique object deleter.
     *
     * Having a default constructor for this class is not ideal, as having an
     * empty deleter here doesn't really make any sense. However, if the
     * deleter class is not default-constructible, then neither is the
     * std::unique_ptr class using it. For that reason, we want it here.
     */
    unique_obj_deleter(void) = default;

    /**
     * @brief Construct a new unique object deleter.
     *
     * This is the preferred constructor for this type. It will store a memory
     * resource, a pointer, a size, and possible an alignment and a number of
     * elements.
     *
     * @param mr The memory resource to use for deallocation.
     * @param p The pointer which we will want to deallocate.
     * @param s The size of the allocation.
     * @param a The alignment of the allocation, with 0 representing no
     * alignment.
     * @param n The number of elements in the allocation.
     */
    unique_obj_deleter(memory_resource& mr, std::size_t s, std::size_t a = 0,
                       std::size_t n = 1)
        : m_mr(&mr), m_size(s), m_align(a), m_elems(n) {}

    /**
     * @brief Copy a unique object deleter.
     *
     * @param i The object deleter to copy.
     */
    unique_obj_deleter(const unique_obj_deleter& i) = default;

    /**
     * @brief Move a unique object deleter.
     *
     * @param i The object deleter to move.
     */
    unique_obj_deleter(unique_obj_deleter&& i) = default;

    /**
     * @brief Copy-assign a unique object deleter.
     *
     * @param i The object to copy into the current one.
     *
     * @return A reference to this object.
     */
    unique_obj_deleter& operator=(const unique_obj_deleter& i) = default;

    /**
     * @brief Move-assign a unique object deleter.
     *
     * @param i The object to move into the current one.
     *
     * @return A reference to this object.
     */
    unique_obj_deleter& operator=(unique_obj_deleter&& i) = default;

    /**
     * @brief Activate the deletion mechanism of the deleter.
     *
     * @param p The pointer to deallocate.
     *
     * @note The pointer argument to this function is in some ways obsolete,
     * because we store the pointer as a member variable. However, having both
     * makes it easier to cross-check the logic.
     */
    void operator()(pointer_t p) const {
        assert(m_mr != nullptr);

        /*
         * If this ever happens, something has gone VERY wrong...
         */
        if (m_mr == nullptr) {
            return;
        }

        /*
         * The class exhibits different behaviour for array types and non-array
         * types.
         */
        if constexpr (!std::is_array_v<T>) {
            /*
             * If we are deleting a non-array type, we can simply call the
             * stored object's destructor.
             */
            p->~T();
        } else if constexpr (std::is_array_v<T>) {
            /*
             * If we are deleting an array type instead, we need to iterate
             * over each of our stored elements and destruct them.
             */
            for (std::size_t i = 0; i < m_elems; ++i) {
                p[i].~storage_t();
            }
        }

        /*
         * Finally, we need to deallocate the memory. For this, we use the
         * memory resource, and we optionally pass it an alignment value.
         */
        if (m_align > 0) {
            m_mr->deallocate(p, m_size, m_align);
        } else {
            m_mr->deallocate(p, m_size);
        }
    };

private:
    /*
     * This should really be a reference, but using a reference here makes the
     * class non-default destructible. At least - without some messy use of
     * default memory resources. Using a pointer here is not ideal, but it
     * allows us to make use of empty unique pointers.
     */
    memory_resource* m_mr;
    std::size_t m_size;
    std::size_t m_align;
    std::size_t m_elems;
};
}  // namespace vecmem::details
