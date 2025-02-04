/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cstddef>
#include <iosfwd>
#include <string>
#include <string_view>

namespace traccc::opts {

/// Common base class for the printable representation of a configuration
class configuration_printable {
    public:
    /// Destructor
    virtual ~configuration_printable();

    /// Return a formatted, printable configuration
    virtual std::string str() const;

    protected:
    friend class configuration_category;
    friend class configuration_list;

    virtual void print_impl(std::ostream& out, std::string_view self_prefix,
                            std::string_view child_prefix,
                            std::size_t prefix_len,
                            std::size_t max_key_width) const = 0;

    virtual std::size_t get_max_key_width_impl() const = 0;

};  // class configuration_printable

}  // namespace traccc::opts
