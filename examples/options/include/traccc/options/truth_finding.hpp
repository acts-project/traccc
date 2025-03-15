/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/common.hpp"
#include "traccc/options/details/interface.hpp"

namespace traccc::opts {
class truth_finding : public interface {

    public:
    float m_min_pt = 0.5f * unit<float>::GeV;
    std::size_t m_min_measurements = 3;

    truth_finding();

    std::unique_ptr<configuration_printable> as_printable() const override;
};
}  // namespace traccc::opts
