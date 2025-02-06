/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "configuration_list.hpp"

// System include(s).
#include <iostream>

namespace traccc::opts {

configuration_list::configuration_list() = default;

configuration_list::~configuration_list() = default;

configuration_list& configuration_list::add_child(child_type child) {

    m_children.push_back(std::move(child));
    return *this;
}

void configuration_list::print_impl(std::ostream& out,
                                    std::string_view self_prefix,
                                    std::string_view child_prefix,
                                    std::size_t prefix_len,
                                    std::size_t max_key_width) const {

    for (std::size_t i = 0; const child_type& child : m_children) {
        child->print_impl(out, self_prefix, child_prefix, prefix_len,
                          max_key_width);
        if (i++ + 1 < m_children.size()) {
            out << '\n';
        }
    }
}

std::size_t configuration_list::get_max_key_width_impl() const {

    std::size_t res = 0;
    for (const child_type& child : m_children) {
        res = std::max(res, child->get_max_key_width_impl());
    }
    return res;
}

}  // namespace traccc::opts
