/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/options/details/interface.hpp"

namespace traccc::opts {

/// Options for memory allocation caching
class allocation_caching
    : public interface {
    public:
    std::size_t m_host_caching_threshold = 2lu << 28lu;
    std::size_t m_device_caching_threshold = 2lu << 28lu;
    
    /// Constructor
    allocation_caching();

    std::unique_ptr<configuration_printable> as_printable() const override;
};
}  // namespace traccc::opts
