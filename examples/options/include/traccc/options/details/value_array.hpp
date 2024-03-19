/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <array>
#include <iosfwd>

namespace traccc::opts {

/// A fixed number of values as one user option.
///
/// @note Implemented as a subclass so it is distinct from `std::array`
///   and we can provide overloads in the same namespace.
///
template <typename TYPE, std::size_t SIZE>
class value_array : public std::array<TYPE, SIZE> {};

/// Extract a fixed number of values from an input of the form 'x:y:z'.
///
/// @note If the values would be separated by whitespace, negative values
///   and additional command line both start with `-` and would be
///   undistinguishable.
///
/// @param is The input stream to read from.
/// @param values The array to fill with the values.
/// @return The input stream for chaining.
///
template <typename TYPE, std::size_t SIZE>
std::istream& operator>>(std::istream& is, value_array<TYPE, SIZE>& values);

/// Print a fixed number of values as `x:y:z`.
///
/// @param os The output stream to write to.
/// @param values The array to print.
/// @return The output stream for chaining.
///
template <typename TYPE, std::size_t SIZE>
std::ostream& operator<<(std::ostream& os,
                         const value_array<TYPE, SIZE>& values);

}  // namespace traccc::opts

// Include the implementation.
#include "value_array.ipp"
