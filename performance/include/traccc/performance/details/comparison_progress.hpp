/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <iostream>
#include <memory>
#include <string_view>

namespace traccc::details {

/// Internal data used by @c comparison_progress
struct comparison_progress_data;

/// Class for tracking the progress of a comparison operation
///
/// It's really just a wrapper around the indicators::progress_bar class.
/// It's here to not expose a public dependency on the indicators library.
///
class comparison_progress {

    public:
    /// Constructor with the total number of steps
    comparison_progress(std::size_t steps, std::ostream& out = std::cout,
                        std::string_view description = "Running comparison ");
    /// Destructor
    ~comparison_progress();

    /// Mark one step done with the progress
    void tick();

    private:
    /// Opaque internal data
    std::unique_ptr<comparison_progress_data> m_data;

};  // class comparison_progress

}  // namespace traccc::details
