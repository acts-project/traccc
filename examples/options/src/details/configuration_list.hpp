/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/options/details/configuration_printable.hpp"

// System include(s).
#include <memory>
#include <vector>

namespace traccc::opts {

/// (Flat) List of configuration parameters
class configuration_list final : public configuration_printable {
    public:
    /// Type for one child of this object
    using child_type = std::unique_ptr<configuration_printable>;
    /// Type for the children of this object
    using children_type = std::vector<child_type>;

    /// Constructor
    explicit configuration_list();
    /// Destructor
    ~configuration_list() final;

    /// Add a single child
    configuration_list& add_child(child_type child);

    private:
    void print_impl(std::ostream& out, std::string_view self_prefix,
                    std::string_view child_prefix, std::size_t prefix_len,
                    std::size_t max_key_width) const final;

    std::size_t get_max_key_width_impl() const final;

    children_type m_children;
};  // class configuration_list

}  // namespace traccc::opts
