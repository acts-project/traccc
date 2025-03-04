/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/options/truth_finding.hpp"

#include <format>

#include "traccc/definitions/common.hpp"
#include "traccc/examples/utils/printable.hpp"

namespace traccc::opts {

truth_finding::truth_finding() : interface("Truth Track Finding Options") {
    m_desc.add_options()(
        "truth-finding-min-pt",
        boost::program_options::value(&m_min_pt)->default_value(
            0.5f * unit<float>::GeV),
        "Candidate particule pT cut [GeV]");
}

std::unique_ptr<configuration_printable> truth_finding::as_printable() const {
    auto cat = std::make_unique<configuration_category>(m_description);

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Minimum pT", std::format("{} GeV", m_min_pt / unit<float>::GeV)));

    return cat;
}
}  // namespace traccc::opts
