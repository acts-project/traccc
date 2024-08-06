/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/utils/helpers.hpp"

// Project include(s).
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/io/event_map2.hpp"

// System include(s).
#include <cmath>
#include <memory>
#include <sstream>

// All in the header file, then will be split.

namespace traccc {
namespace details {

struct verbose_performance_metrics {

    verbose_performance_metrics(bool is_ambiguity_resolution_,
                                const std::string& algorithm_name_ = "")
        : algorithm_name(algorithm_name_),
          is_ambiguity_resolution(is_ambiguity_resolution_) {}

    /**
     * @brief
     *
     * @param all_tracks Measurements for each track
     * @param selected_tracks Indexes of the tracks selected by the ambiguity
     * resolution algorithm
     */
    void ambiguity_resolution(
        std::vector<std::vector<measurement>> const& all_tracks,
        const std::vector<std::size_t>& selected_tracks,
        const event_map2& evt_map);

    void ambiguity_resolution(
        const track_state_container_types::host& all_tracks,
        const std::vector<std::size_t>& selected_tracks,
        const event_map2& evt_map);

    void generic(const track_candidate_container_types::host& all_tracks,
                 const event_map2& evt_map);

    void generic(const track_state_container_types::host& all_tracks,
                 const event_map2& evt_map);

    void generic(const std::vector<std::vector<measurement>>& all_tracks,
                 const event_map2& evt_map);

    std::ostream& print(std::ostream& os);

    // For seeding, finding, and fitting algorithms
    struct item_t {
        std::size_t valid = 0, duplicate = 0, fake = 0;

        // Values computed by compute_percentages
        std::size_t valid_p = 0, duplicate_p = 0, fake_p = 0;

        void compute_percentages() {
            double total = valid + duplicate + fake;
            if (total > 0) {
                valid_p = std::round(100 * valid / total);
                duplicate_p = std::round(100 * duplicate / total);
                fake_p = std::round(100 * fake / total);
            } else {
                valid_p = 0;
                duplicate_p = 0;
                fake_p = 0;
            }
        }
    };

    // Only for ambiguity resolution algorithm
    struct ar_t {
        // Among selected tracks
        item_t selected;

        // std::size_t sel_valid = 0, sel_duplicate = 0, sel_fake = 0;
        // Among evicted tracks
        item_t evicted;
        // std::size_t del_valid = 0, del_duplicate = 0, del_fake = 0;

        // The number of times the ambiguity_resolution function was called
        std::size_t call_count = 0;
        // The sum of the results quality for valid tracks
        double valid_quality_sum = 0;
    };

    private:
    ar_t ar_metrics;
    item_t gen_metrics;
    std::string algorithm_name;
    bool is_ambiguity_resolution;
    // std::size_t call_count = 0;
};
}  // namespace details
}  // namespace traccc
