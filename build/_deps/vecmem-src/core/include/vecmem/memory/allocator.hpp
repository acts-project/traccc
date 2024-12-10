/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <cstddef>

namespace vecmem {

/**
 * @brief An allocator class that wraps a memory resource.
 *
 * Sometimes we want to construct objects outside of vectors, and for those
 * cases we need a simpler allocator that uses a memory resource to set the
 * allocation strategy. This class is inspired by
 * @c std::pmr::polymorphic_allocator but changes some things about it.
 * Firstly, this class is not templated at a class level, and the templating
 * only happens at a method level. This makes the class significantly easier
 * to use. Secondly, this class emulates some of the C++20 functionality
 * offered by the polymorphic allocator in a C++17 context.
 *
 * @warning Deallocating memory allocated with a different allocator (or
 * rather, with an allocator using a different upstream memory resource)
 * than the one doing the deallocation is not well-defined and should be
 * avoided.
 */
class VECMEM_CORE_EXPORT allocator {
public:
    /**
     * @brief Construct an allocator.
     *
     * Construct a new allocator on a given upstream memory resource.
     *
     * @param[in] mem The memory resource to use for allocations.
     */
    allocator(memory_resource& mem);

    /**
     * @brief Allocate a given number of bytes.
     *
     * The most low-level allocation method provided by the allocator simply
     * allocates a number of untyped bytes, optionally with some alignment.
     * This can be used directly, or by other allocation methods to build
     * more high-level functionality.
     *
     * @param[in] bytes The number of bytes to allocate.
     * @param[in] alignment The alignment boundary for the allocation.
     */
    void* allocate_bytes(std::size_t bytes,
                         std::size_t alignment = alignof(std::max_align_t));

    /**
     * @brief Deallocate a given number of bytes.
     *
     * The most low-level deallocation method simply deallocates a number of
     * bytes.
     *
     * @warning The onus is on the user to remember the size of each
     * allocation, and this method might be too low-level for most purposes.
     *
     * @param[in] p The pointer to deallocate.
     * @param[in] bytes The number of bytes to deallocate.
     * @param[in] alignment The alignment boundary for the deallocation.
     */
    void deallocate_bytes(void* p, std::size_t bytes,
                          std::size_t alignment = alignof(std::max_align_t));

    /**
     * @brief Allocate space for (a number of) objects.
     *
     * A mid-level allocator which abstracts away calculating the size of
     * their allocation. The user simply provides the type and the number
     * of objects to allocate space for.
     *
     * @note The space allocated by this method is not initialized in any
     * way.
     *
     * @tparam T The type of object to allocate space for.
     * @param[in] n The number of objects of type T to allocate space for.
     */
    template <typename T>
    T* allocate_object(std::size_t n = 1);

    /**
     * @brief Deallocate space for (a number of) objects.
     *
     * A mid-level deallocator which abstracts away calculating the size of
     * the deallocation.
     *
     * @note The space deallocated by this method is not deconstructed.
     *
     * @tparam T The type of object to deallocater.
     * @param[in] n The number of objects of type T to deallocate.
     */
    template <typename T>
    void deallocate_object(T* p, std::size_t n = 1);

    /**
     * @brief Allocate and construct a new object.
     *
     * The highest-level allocator we provide, this allocates memory for and
     * then constructs an object of the given type. Parameters passed to
     * this method are passed on to the constructor for the object.
     *
     * @tparam T The type to construct.
     * @tparam Args The constructor parameter pack.
     * @param[in] args The arguments to pass on to the class constructor.
     */
    template <typename T, typename... Args>
    T* new_object(Args&&... args);

    /**
     * @brief Deconstruct and deallocate an object.
     *
     * The highest-level deallocator we provide, this deconstructs and then
     * deallocates an object.
     *
     * @tparam T The type of object to delete.
     * @param[in] p The object to delete.
     */
    template <typename T>
    void delete_object(T* p);

private:
    memory_resource& m_mem;

};  // class allocator

}  // namespace vecmem

// Include the implementation.
#include "vecmem/memory/impl/allocator.ipp"
