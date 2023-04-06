/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/algorithm/string.hpp>

// System include(s).
#include <array>
#include <iosfwd>
#include <istream>
#include <optional>
#include <ostream>
#include <vector>

namespace traccc {

namespace {
static constexpr std::string_view s_separator = ":";
}

/// A fixed number of real values as one user option.
///
/// @note Implemented as a subclass so it is distinct from `std::array`
///   and we can provide overloads in the same namespace.
template <typename scalar_t, size_t kSize>
class Reals : public std::array<scalar_t, kSize> {};

namespace detail {

template <typename value_t, typename converter_t>
void parse_variable(std::istream& is, std::vector<value_t>& values,
                    converter_t&& convert) {
    values.clear();

    std::string buf;
    is >> buf;
    std::vector<std::string> stringValues;
    boost::split(stringValues, buf, boost::is_any_of(s_separator));
    for (const std::string& stringValue : stringValues) {
        values.push_back(convert(stringValue));
    }
}

template <typename value_t, typename converter_t>
void parse_fixed(std::istream& is, size_t size, value_t* values,
                 converter_t&& convert) {
    // reserve space for the expected number of values
    std::vector<value_t> tmp(size, 0);
    parse_variable(is, tmp, std::forward<converter_t>(convert));
    if (tmp.size() < size) {
        throw std::invalid_argument(
            "Not enough values for fixed-size user option, expected " +
            std::to_string(size) + " received " + std::to_string(tmp.size()));
    }
    if (size < tmp.size()) {
        throw std::invalid_argument(
            "Too many values for fixed-size user option, expected " +
            std::to_string(size) + " received " + std::to_string(tmp.size()));
    }
    std::copy(tmp.begin(), tmp.end(), values);
}

}  // namespace detail

/// Extract a fixed number of doubles from an input of the form 'x:y:z'.
///
/// @note If the values would be separated by whitespace, negative values
///   and additional command line both start with `-` and would be
///   undistinguishable.
template <typename scalar_t, size_t kSize>
inline std::istream& operator>>(std::istream& is,
                                Reals<scalar_t, kSize>& values) {
    detail::parse_fixed(is, kSize, values.data(),
                        [](const std::string& s) { return std::stod(s); });

    return is;
}

/// Print a fixed number of doubles as `x:y:z`.
template <typename scalar_t, size_t kSize>
inline std::ostream& operator<<(std::ostream& os,
                                const Reals<scalar_t, kSize>& values) {

    const auto data = values.data();

    for (size_t i = 0; i < kSize; ++i) {
        if (0u < i) {
            os << s_separator;
        }
        os << data[i];
    }

    return os;
}

}  // namespace traccc