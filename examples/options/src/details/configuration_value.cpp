/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "configuration_value.hpp"

// System include(s).
#include <format>
#include <iostream>

namespace traccc::opts {

configuration_value::configuration_value(std::string_view key,
                                         std::string_view value)
    : m_key{key}, m_value{value} {}

configuration_value::~configuration_value() = default;

void configuration_value::print_impl(std::ostream& out,
                                     std::string_view self_prefix,
                                     std::string_view, std::size_t prefix_len,
                                     std::size_t max_key_width) const {

    out << std::format(
        "{} {}: {}{}\n", self_prefix, m_key,
        std::string(max_key_width - prefix_len - m_key.length(), ' '), m_value);
}

std::size_t configuration_value::get_max_key_width_impl() const {
    return m_key.length();
}

}  // namespace traccc::opts
