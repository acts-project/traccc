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
#include <string>
#include <string_view>
#include <vector>

namespace traccc::opts {

/// A printable category of other printable objects
class configuration_category final : public configuration_printable {
    public:
    /// Type for one child of this object
    using child_type = std::unique_ptr<configuration_printable>;
    /// Type for the children of this object
    using children_type = std::vector<child_type>;

    /// Constructor
    explicit configuration_category(std::string_view name);
    /// Destructor
    ~configuration_category() override;

    /// Add a single child
    configuration_category& add_child(child_type child);

    private:
    void print_impl(std::ostream& out, std::string_view self_prefix,
                    std::string_view child_prefix, std::size_t prefix_len,
                    std::size_t max_key_width) const override;

    std::size_t get_max_key_width_impl() const override;

    std::string m_name;
    children_type m_children;
};  // class configuration_category

}  // namespace traccc::opts
