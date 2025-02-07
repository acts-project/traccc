/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <string>
#include "traccc/options/allocation_caching.hpp"
#include "traccc/examples/utils/printable.hpp"

namespace {
std::string pretty_print_bytes(std::size_t n) {
    if (n % (1024lu * 1024lu * 1024lu) == 0) {
        return std::to_string(n / (1024lu * 1024lu * 1024lu)) + " GiB";
    } else if (n % (1024lu * 1024lu) == 0) {
        return std::to_string(n / (1024lu * 1024lu)) + " MiB";
    } else if (n % 1024lu == 0) {
        return std::to_string(n / 1024lu) + " KiB";
    } else {
        return std::to_string(n) + " B";
    }
}
}

namespace traccc::opts {

allocation_caching::allocation_caching() : interface("Memory Allocation Options") {
    m_desc.add_options()("host-caching-threshold",
                         boost::program_options::value(&m_host_caching_threshold)->default_value(m_host_caching_threshold),
                         "Threshold (in bytes) below which to cache host allocations");
    m_desc.add_options()("device-caching-threshold",
                         boost::program_options::value(&m_device_caching_threshold)->default_value(m_device_caching_threshold),
                         "Threshold (in bytes) below which to cache device allocations");
}

std::unique_ptr<configuration_printable> allocation_caching::as_printable() const {
    auto cat = std::make_unique<configuration_category>("Memory allocation options");

    cat->add_child(
        std::make_unique<configuration_kv_pair>(
            "Host caching threshold", pretty_print_bytes(m_host_caching_threshold)));
    cat->add_child(
        std::make_unique<configuration_kv_pair>(
            "Device caching threshold", pretty_print_bytes(m_device_caching_threshold)));

    return cat;
}

}  // namespace traccc::opts
