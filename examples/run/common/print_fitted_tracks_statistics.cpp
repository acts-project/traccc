/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "print_fitted_tracks_statistics.hpp"

namespace traccc::details {

void print_fitted_tracks_statistics(
    const edm::track_fit_container<default_algebra>::host& tracks,
    const Logger& log) {

    std::size_t success = 0;
    std::size_t non_positive_ndf = 0;
    std::size_t not_all_smoothed = 0;

    for (track_fit_outcome outcome : tracks.tracks.fit_outcome()) {
        if (outcome == track_fit_outcome::SUCCESS) {
            ++success;
        } else if (outcome == track_fit_outcome::FAILURE_NON_POSITIVE_NDF) {
            ++non_positive_ndf;
        } else if (outcome == track_fit_outcome::FAILURE_NOT_ALL_SMOOTHED) {
            ++not_all_smoothed;
        }
    }

    auto logger = [&log]() -> const Logger& { return log; };
    TRACCC_INFO("Success: " << success
                            << "  Non positive NDF: " << non_positive_ndf
                            << "  Not all smoothed: " << not_all_smoothed
                            << "  Total: " << tracks.tracks.size());
}

}  // namespace traccc::details
