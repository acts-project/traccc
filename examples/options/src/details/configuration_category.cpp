/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "configuration_category.hpp"

// System include(s).
#include <algorithm>
#include <format>
#include <iostream>

namespace traccc::opts {

configuration_category::configuration_category(std::string_view name)
    : m_name{name} {}

configuration_category::~configuration_category() = default;

configuration_category& configuration_category::add_child(child_type child) {

    m_children.push_back(std::move(child));
    return *this;
}

void configuration_category::print_impl(std::ostream& out,
                                        std::string_view self_prefix,
                                        std::string_view child_prefix,
                                        std::size_t prefix_len,
                                        std::size_t max_key_width) const {

    out << std::format("{}┬ {}\n", self_prefix, m_name);
    if (m_children.empty()) {
        out << std::format("{}└─ <empty>\n", child_prefix);
    } else {
        for (std::size_t i = 0; const child_type& child : m_children) {
            if (i == m_children.size() - 1) {
                child->print_impl(out, std::format("{}└─", child_prefix),
                                  std::format("{}  ", child_prefix),
                                  prefix_len + 2, max_key_width);
            } else {
                child->print_impl(out, std::format("{}├─", child_prefix),
                                  std::format("{}│ ", child_prefix),
                                  prefix_len + 2, max_key_width);
            }
            ++i;
        }
    }
}

std::size_t configuration_category::get_max_key_width_impl() const {

    std::size_t res = 0;
    for (const child_type& child : m_children) {
        res = std::max(res, 2 + child->get_max_key_width_impl());
    }
    return res;
}

}  // namespace traccc::opts
