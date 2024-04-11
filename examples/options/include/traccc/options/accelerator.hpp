/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/options/details/interface.hpp"

namespace traccc::opts {

/// Option(s) for accelerator usage
class accelerator : public interface {

    public:
    /// @name Options
    /// @{

    /// Whether to compare the accelerator code's output with that of the CPU
    bool compare_with_cpu = false;

    /// @}

    /// Constructor
    accelerator();

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // struct accelerator

}  // namespace traccc::opts
