/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstddef>

namespace vecmem {
template <typename T>
T *allocator::allocate_object(std::size_t n) {
    /*
     * Since we know the object being allocated here, we can multiply the
     * number of objects to allocate with the size of that class. We can
     * also use the compiler's knowledge about the alignment requirements of
     * the class.
     */
    return static_cast<T *>(m_mem.allocate(n * sizeof(T), alignof(T)));
}

template <typename T>
void allocator::deallocate_object(T *p, std::size_t n) {
    /*
     * Use the upstream allocator to deallocate the memory, again using the
     * compiler's knowledge about the size and the alignment requirements of
     * the class that is to be deallocated.
     */
    m_mem.deallocate(p, n * sizeof(T), alignof(T));
}

template <typename T, typename... Args>
T *allocator::new_object(Args &&... args) {
    /*
     * Firstly we need to allocate some space for the object which we are
     * constructing. We use the allocate_object method for this, with the
     * default argument of 1 for a single object.
     */
    void *p = allocate_object<T>();

    /*
     * Next, we use the placement new operation to construct the object in
     * the memory which we have just allocated. For this, we also need to
     * forward the arguments passed to this method so they can be used to
     * call the constructor of the class.
     */
    return new (p) T(std::forward<Args>(args)...);
}

template <typename T>
void allocator::delete_object(T *p) {
    /*
     * Before ruthlessly destroying the object, we will give it a few last
     * words to deconstruct itself. This is to ensure it can safely
     * do any clean-up it needs to do.
     */
    p->~T();

    /*
     * Once the deallocation is complete, we can get rid of the memory as
     * well. For this, we call the higher-level deallocate_object method.
     */
    deallocate_object<T>(p);
}
}  // namespace vecmem
