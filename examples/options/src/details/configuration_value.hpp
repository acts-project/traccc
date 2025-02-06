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
#include <string>
#include <string_view>

namespace traccc::opts {

/// Configuration value (key-value pair)
class configuration_value final : public configuration_printable {
    public:
    /// Constructor
    explicit configuration_value(std::string_view key, std::string_view value);
    /// Destructor
    ~configuration_value() override;

    private:
    void print_impl(std::ostream& out, std::string_view self_prefix,
                    std::string_view, std::size_t prefix_len,
                    std::size_t max_key_width) const override;

    std::size_t get_max_key_width_impl() const override;

    std::string m_key;
    std::string m_value;
};  // class configuration_value

}  // namespace traccc::opts
