/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/performance.hpp"

#include "details/configuration_category.hpp"
#include "details/configuration_value.hpp"

// System include(s).
#include <format>

namespace traccc::opts {

performance::performance() : interface("Performance Measurement Options") {

    m_desc.add_options()("check-performance",
                         boost::program_options::bool_switch(&run),
                         "Run performance checks");
}

std::unique_ptr<configuration_printable> performance::as_printable() const {

    auto result = std::make_unique<configuration_category>(m_description);

    result->add_child(std::make_unique<configuration_value>(
        "Run performance checks", std::format("{}", run)));

    return result;
}
}  // namespace traccc::opts
