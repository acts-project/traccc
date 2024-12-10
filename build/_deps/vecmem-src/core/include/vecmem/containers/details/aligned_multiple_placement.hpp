/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// vecmem headers
#include "vecmem/memory/unique_ptr.hpp"

// Standard library headers
#include <tuple>
#include <type_traits>

namespace vecmem {
namespace details {
/**
 * @brief Allocation of aligned arrays of given types.
 *
 * This function allocates a chunk of memory containing a set of arrays of
 * different types in a consecutive fashion. For example, let's say we want to
 * allocate:
 *
 * 1. 3 ints (4 bytes each, aligned to 4 byte boundaries)
 * 2. 2 longs (8 bytes each, aligned to 8 byte boundaries)
 * 3. 7 shorts (2 bytes each, unaligned)
 *
 * This corresponds with the following evocation of this function:
 *
 *     aligned_multiple_placement<int, long, short>(r, 3, 2, 7)
 *
 * Under these rules, the function would allocate a chunk of memory that looks
 * like this:
 *
 *          ┌────────────────────────────────────────────────────────────────┐
 *     Byte │╭ 00      ╭ 10      ╭ 20      ╭ 30      ╭ 40      ╭ 50      ╭ 60│
 *     Data │-----###IIIIIIIIIIII####LLLLLLLLLLLLLLLLSSSSSSSSSSSSSS###-------│
 *     Info │╰ A  ╰ B╰ C             ╰ D             ╰ E             ╰ F     │
 *          └────────────────────────────────────────────────────────────────┘
 *
 * In this diagram, we see a small 64-byte memory space, with the "-" character
 * denoting memory that is unallocated, "#" denoting memory that is used as
 * padding to meet alignment requirements, and "I", "L", and "S" denoting ints,
 * longs, and shorts, respectively.
 *
 * The total memory space runs from marker A to past the end of the diagram.
 * The region from B to F indicates the memory that we have allocated to store
 * our data. Note that we do not have control over where this region starts! In
 * other words, A could be positioned anywhere along the memory space. In
 * reality, most allocators have their own alignment guarantees and would not
 * return memory at address 5 like we see here. However, we include it here for
 * demonstrative purposes.
 *
 * Marker C denotes the start of our region of three integers, which takes
 * twelve bytes in total. The important idea here is that we cannot start this
 * region right at the start of our allocation (at B), because address B, 5, is
 * not a multiple of the 4-byte alignment requirement of the integer type.
 * Thus, we must insert three bytes of padding (denoted by "#"), and our ints
 * start at marker C (address 8). They end twelve bytes later, at address 20.
 *
 * Following our integers, we want to allocate two longs, which must be aligned
 * to 8 byte boundaries. If we were to immediately succeed our integers with
 * these longs, they would start at address 20, which is not a multiply of 8.
 * Therefore, we must insert additional padding before starting our region of
 * longs at marker D (address 24).
 *
 * It is worth noting that this method extends to an arbitrary number of type
 * regions, although in practice the number will usually be limited (it might
 * even just be 2 in all cases). If there is only a single type, this function
 * simply serves as an alignment operator akin to `std::align`.
 *
 * @note This function is pessimistic in its allocation of memory, meaning that
 * it will usually allocate slightly too much memory to be on the safe side.
 * This should have a negligible impact on real-world applications.
 *
 * @tparam Ts The types to allocate memory for.
 * @tparam Ps Positional parameter types, which should all be std::size or a
 * compatible type, and the length must be equal to that of `Ts`.
 * @param r The memory resource to use for allocation.
 * @param ps The set of sizes of the differently typed regions.
 * @returns A tuple of |Ts| + 1 elements, where the first element is a unique
 * allocation pointer to the complete block of memory (such that it may be
 * freed), and the other values represent the addresses in memory where the
 * regions begin.
 */
template <typename... Ts, typename... Ps>
std::tuple<vecmem::unique_alloc_ptr<char[]>, std::add_pointer_t<Ts>...>
aligned_multiple_placement(vecmem::memory_resource &r, Ps &&... ps);
}  // namespace details
}  // namespace vecmem

#include "vecmem/containers/impl/aligned_multiple_placement.ipp"
