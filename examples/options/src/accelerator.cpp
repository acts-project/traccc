/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/accelerator.hpp"

#include "traccc/examples/utils/printable.hpp"

// System include(s).
#include <format>

namespace traccc::opts {

accelerator::accelerator() : interface("Accelerator Options") {

    m_desc.add_options()("compare-with-cpu",
                         boost::program_options::bool_switch(&compare_with_cpu),
                         "Compare accelerator output with that of the CPU");
}

std::unique_ptr<configuration_printable> accelerator::as_printable() const {
    auto cat = std::make_unique<configuration_category>(m_description);

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Compare with CPU output", std::format("{}", compare_with_cpu)));

    return cat;
}

}  // namespace traccc::opts
