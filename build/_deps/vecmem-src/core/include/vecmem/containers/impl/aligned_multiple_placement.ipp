/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// vecmem headers
#include "vecmem/utils/type_traits.hpp"

// Standard library headers
#include <algorithm>
#include <tuple>
#include <type_traits>

namespace vecmem {
namespace details {
/**
 * @brief Helper function for aligned_multiple_placement.
 *
 * First, please don't use this function by itself. It is meant to be used only
 * as a helper function.
 *
 * This function recursively reserves regions in a pre-allocated region of
 * memory, and enforces most of the alignment semantics that are given for the
 * aligned_multiple_placement function.
 *
 * @warning This function enforces no requirements on anything, because those
 * requirements are already enforced by aligned_multiple_placement.
 *
 * @tparam T The head of the pack of types.
 * @tparam Ts The (possibly empty) tail of the pack of types.
 * @tparam P The head of the pack of sizes.
 * @tparam Ps The (possibly empty) tail of the pack of sizes.
 * @param p The pointer at which (or past which) to begin this region.
 * @param q The remaining space available for allocation.
 * @param n The number of objects of type T to allocate space for.
 * @param Ps The remaining sizes for future types.
 * @returns A tuple of size |Ts| + 1, where each element is a pointer to the
 * address that marks the beginning of those regions.
 */
template <typename T, typename... Ts, typename P, typename... Ps>
std::tuple<std::add_pointer_t<T>, std::add_pointer_t<Ts>...>
aligned_multiple_placement_helper(void *p, std::size_t q, P n, Ps &&... ps) {
    /*
     * We start out by calculating the size of the current region.
     */
    std::size_t size = sizeof(T) * static_cast<std::size_t>(n);

    /*
     * We will start out by having this region point to null, because it is
     * possible to not allocate anything: this is the case if n is equal to
     * zero, and we support this for convenience if this happens
     * programmatically.
     */
    T *beg = nullptr;

    /*
     * As mentioned, we only proceed to allocation if we actually need any
     * memory for this region.
     */
    if (size > 0) {
        /*
         * We will use `std::align` to determine the beginning of this region
         * of memory. This function has a few side-effects. The returned
         * pointer is aligned to the boundary of T, and:
         *
         * 1. The variable p is updated such that it equals beg.
         * 2. The size is decreased to compensate for the side of the padding,
         *    but crucially NOT for the size of the allocation itself!
         */
        beg = static_cast<T *>(std::align(alignof(T), size, p, q));

        /*
         * If `std::align` returns a null pointer, there was insufficient
         * memory to find an allocated pointer. This should never happen in
         * practice, but we should check for it regardless.
         */
        assert(beg != nullptr);

        /*
         * Now, we will update our current pointer and remaining size (which,
         * remember, `std::align` does not do: it only gives the start of the
         * region and reduces the size by the padding). Essentially, this gives
         * us the parameters for the next invocation of this function.
         */
        p = static_cast<void *>(static_cast<char *>(p) + size);
        q -= size;
    }

    /*
     * We now proceed in one of two different ways: if there are any remaining
     * regions to reserve memory for, we call this function recursively. Note
     * that the base case of this recursion is static and known at compile
     * time. If we have no more types to reserve for, then we are done.
     */
    if constexpr (sizeof...(Ts) > 0) {
        /*
         * We apply here the same tuple concatenation trick that we use in
         * aligned_multiple_placement: we concate a single-element tuple
         * containing the start of the current region with whatever is returned
         * by the recursive invocation of this method.
         */
        return std::tuple_cat(
            /*
             * Here we construct the single-element tuple.
             */
            std::make_tuple<std::add_pointer_t<T>>(std::move(beg)),
            /*
             * Next, we recursively call this method, but with one type
             * argument stripped, as well as one size argument. We use the
             * updated pointer and remaining size for this.
             */
            aligned_multiple_placement_helper<Ts...>(p, q,
                                                     std::forward<Ps>(ps)...));
    } else {
        /*
         * If we have no more types, we can return a single-element tuple. This
         * tuple will be concatenated into a larger one by the caller.
         */
        return std::make_tuple<std::add_pointer_t<T>>(std::move(beg));
    }
#ifdef __GNUC__
    // Certain combinations of CUDA + GCC generate a warning here, thinking
    // that the code may reach this point in the function. So for just GCC,
    // let's add some help here. Telling it that this part of the function is
    // not (meant to be) reachable.
    __builtin_unreachable();
#endif  // __GNUC__
}

template <typename... Ts, typename... Ps>
std::tuple<vecmem::unique_alloc_ptr<char[]>, std::add_pointer_t<Ts>...>
aligned_multiple_placement(vecmem::memory_resource &r, Ps &&... ps) {
    /*
     * First, we will assert that we have exactly as many template arguments as
     * we have positional arguments, barring the memory resource. This is very
     * important, because each template type must be accompanied by a number of
     * times that type should occur.
     */
    static_assert(sizeof...(Ts) == sizeof...(Ps),
                  "Number of type parameters must be equal to the number of "
                  "value parameters.");

    /*
     * Next, we want all of our positional arguments to be size types, or at
     * least some type that is trivially convertible to it. We can't really
     * enforce that cleanly using C++, so we just accept a parameter pack of
     * arbitrary types and then we check that they're all size-like.
     */
    static_assert(std::conjunction_v<std::is_convertible<Ps, std::size_t>...>,
                  "All parameters must be of type std::size_t");

    /*
     * Calculate the maximum alignment between all of the types in the template
     * parameter pack, which will determine the amount of additional space
     * we need to allocate.
     */
    std::size_t alignment = vecmem::details::max(alignof(Ts)...);

    /*
     * Next, we pessimistically calculate the number of bytes we need. We do
     * this by considering all the types in our parameter pack and multiplying
     * their size by the number of times we want each type to exist. That
     * happens using the C++ template fold expression. Then, we add some
     * buffer, because each additional type will risk wasting some space due to
     * alignment requirements. We do this by adding the maximum alignment once
     * for each type, because each type introduces at _most_ that amount of
     * wasted space. This guarantees that we never run out of space, but it can
     * sometimes waste a little bit of space (on the order of dozens of bytes
     * for the common case, but potentially up to a few kilobytes for severely
     * over-aligned types).
     */
    const std::size_t bytes =
        ((static_cast<std::size_t>(ps) * sizeof(Ts)) + ...) +
        sizeof...(Ts) * alignment;

    /*
     * We can now make the allocation request with the memory resource, which
     * in this case does not need to know anything about our alignment
     * requirements: we are going to handle all of those ourselves.
     */
    vecmem::unique_alloc_ptr<char[]> ptr =
        vecmem::make_unique_alloc<char[]>(r, bytes);

    /*
     * We'll make a non-owning copy of this pointer, because we will need it in
     * a moment _after_ we've moved it, so we cannot use the original version
     * of it anymore.
     */
    void *p = ptr.get();

    /*
     * Now we come to the meat of the pudding. Remember that we want to return
     * a tuple containing a unique pointer to the start of the allocation as
     * a whole, and then one pointer for where each of our type regions starts.
     * To accomplish this, we will concatenate two smaller tuples, namely:
     *
     * 1. A single-element tuple containing the unique pointer to the start of
     *    the allocated region.
     * 2. A tuple of length equal to the number of type parameters, each
     *    element of which is a pointer representing where that type's region
     *    starts.
     *
     * It's a little bit silly that we need to construct a single-element tuple
     * for the first part, but that's simply how `std::tuple_cat` works.
     */
    return std::tuple_cat(
        /*
         * Here, we construct our single-element tuple, into which we simply
         * move the unique pointer which manages the memory that we have
         * allocated previously.
         */
        std::make_tuple<vecmem::unique_alloc_ptr<char[]>>(std::move(ptr)),
        /*
         * And here we call a helper function, which will actually do all of
         * the alignment. We pass it the start of our allocation, the amount of
         * remaining space, and then the number of times we want all of our
         * types to appear.
         */
        aligned_multiple_placement_helper<Ts...>(p, bytes,
                                                 std::forward<Ps>(ps)...));
}
}  // namespace details
}  // namespace vecmem
