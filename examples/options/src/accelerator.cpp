/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/accelerator.hpp"

#include "details/configuration_category.hpp"
#include "details/configuration_value.hpp"

// System include(s).
#include <format>

namespace traccc::opts {

accelerator::accelerator() : interface("Accelerator Options") {

    m_desc.add_options()("compare-with-cpu",
                         boost::program_options::bool_switch(&compare_with_cpu),
                         "Compare accelerator output with that of the CPU");
}

std::unique_ptr<configuration_printable> accelerator::as_printable() const {

    auto result = std::make_unique<configuration_category>(m_description);

    result->add_child(std::make_unique<configuration_value>(
        "Compare with CPU output", std::format("{}", compare_with_cpu)));

    return result;
}

}  // namespace traccc::opts
