/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/performance/details/comparison_progress.hpp"

// Indicators include(s).
#include <indicators/progress_bar.hpp>

namespace traccc::details {

struct comparison_progress_data {

    /// Progress bar used internally
    std::unique_ptr<indicators::ProgressBar> m_bar;

};  // struct comparison_progress_data

comparison_progress::comparison_progress(std::size_t steps, std::ostream& out,
                                         std::string_view description)
    : m_data(std::make_unique<comparison_progress_data>()) {

    // Set up the progress bar.
    m_data->m_bar = std::make_unique<indicators::ProgressBar>(
        indicators::option::BarWidth{50},
        indicators::option::PrefixText{description.data()},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::MaxProgress{steps},
        indicators::option::Stream{out});
}

comparison_progress::~comparison_progress() = default;

void comparison_progress::tick() {

    m_data->m_bar->tick();
}

}  // namespace traccc::details
