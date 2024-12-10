/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cassert>
#include <climits>
#include <cstddef>
#include <limits>
#ifdef VECMEM_HAVE_LZCNT_U64
#include <intrin.h>
#endif

namespace vecmem::details {

/// Check if a number is a power of 2
///
/// @param x The number to check
/// @return @c true if @c x is a power of 2, @c false otherwise
///
inline bool is_power_of_2(std::size_t x) {

    return ((x & (x - 1)) == static_cast<std::size_t>(0));
}

/// Count the leading zeroes in a number
///
/// @param i The number to count the leading zeroes in
/// @return The number of leading zeroes in @c i
///
inline std::size_t clzl(std::size_t i) {

#if defined(VECMEM_HAVE_LZCNT_U64)
    return static_cast<std::size_t>(_lzcnt_u64(i));
#elif defined(VECMEM_HAVE_BUILTIN_CLZL)
    return static_cast<std::size_t>(__builtin_clzl(i));
#else
    std::size_t b;
    for (b = 0;
         !((i << b) & (static_cast<std::size_t>(1UL)
                       << (std::numeric_limits<std::size_t>::digits - 1UL)));
         ++b)
        ;
    return b;
#endif
}

/// Compute the base-2 logarithm of a number
///
/// @param x The number to compute the logarithm of
/// @return The base-2 logarithm of @c x
///
inline std::size_t log2(std::size_t x) {

    static constexpr std::size_t num_bits = CHAR_BIT * sizeof(std::size_t);
    static constexpr std::size_t num_bits_minus_one = num_bits - 1;
    return num_bits_minus_one - clzl(x);
}

/// Compute the base-2 logarithm of a number, rounding up to the nearest log
///
/// @param x The number to compute the logarithm of
/// @return The base-2 logarithm of @c x
///
inline std::size_t log2_ri(std::size_t x) {

    // Compute the log.
    std::size_t result = log2(x);

    // Round up to the nearest log.
    if (!is_power_of_2(x)) {
        ++result;
    }

    // Return the result.
    return result;
}

/**
 * @brief Rounds a size up to the nearest power of two, and returns the power
 * (not the size itself).
 */
inline std::size_t round_up(std::size_t size) {
    static constexpr std::size_t SIZE_T_BITS = CHAR_BIT * sizeof(std::size_t);
    assert((static_cast<std::size_t>(1UL) << (SIZE_T_BITS - 1)) >= size);
    for (std::size_t i = 0; i < SIZE_T_BITS; ++i) {
        if ((static_cast<std::size_t>(1UL) << i) >= size) {
            return i;
        }
    }

    return 0;
}

}  // namespace vecmem::details
