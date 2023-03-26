/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <memory>
#include <optional>
#include <traccc/efficiency/nseed_performance_writer.hpp>
#include <traccc/efficiency/track_filter.hpp>
#include <traccc/efficiency/track_matcher.hpp>

namespace traccc {
nseed_performance_writer::nseed_performance_writer(
    const std::string& prefix, std::unique_ptr<track_filter>&& filter,
    std::unique_ptr<track_matcher>&& matcher)
    : _prefix(prefix),
      _filter(std::forward<std::unique_ptr<track_filter>>(filter)),
      _matcher(std::forward<std::unique_ptr<track_matcher>>(matcher)) {}

void nseed_performance_writer::finalize() {
    output_seed_file.close();
}

void nseed_performance_writer::write_seed_header() {
    output_seed_file << "event_id" << sep << "seed_id" << sep << "length" << sep
                     << "particle_id" << std::endl;
}

void nseed_performance_writer::write_seed_row(
    std::size_t event_id, std::size_t seed_id, std::size_t length,
    std::optional<std::size_t> particle_id) {
    output_seed_file << event_id << sep << seed_id << sep << length << sep
                     << (particle_id ? std::to_string(*particle_id) : "")
                     << std::endl;
}

std::string nseed_performance_writer::generate_report_str() const {
    char buffer[512];
    std::string result;

    snprintf(buffer, 512, "==> Seed finding efficiency ...\n");
    result.append(buffer);

    snprintf(buffer, 512, "- %-20s : %s\n", "Particle filter",
             _filter->get_name().c_str());
    result.append(buffer);
    snprintf(buffer, 512, "- %-20s : %s\n", "Particle matcher",
             _matcher->get_name().c_str());
    result.append(buffer);

    snprintf(buffer, 512, "- %-20s : %6ld\n", "Total seeds",
             (_stats.true_seeds + _stats.false_seeds));
    result.append(buffer);
    snprintf(buffer, 512, "- %-20s : %6ld\n", "True seeds", _stats.true_seeds);
    result.append(buffer);
    snprintf(buffer, 512, "- %-20s : %6ld\n", "False seeds",
             _stats.false_seeds);
    result.append(buffer);
    snprintf(buffer, 512, "- %-20s : %6ld\n", "Total tracks",
             (_stats.unmatched_tracks + _stats.matched_tracks));
    result.append(buffer);
    snprintf(buffer, 512, "- %-20s : %6ld\n", "Matched tracks",
             _stats.matched_tracks);
    result.append(buffer);
    snprintf(buffer, 512, "- %-20s : %6ld\n", "Unmatched tracks",
             _stats.unmatched_tracks);
    result.append(buffer);

    snprintf(buffer, 512, "- %-20s : %6.2f%%\n", "Precision",
             (100. * static_cast<double>(_stats.true_seeds) /
              static_cast<double>(_stats.true_seeds + _stats.false_seeds)));
    result.append(buffer);
    snprintf(buffer, 512, "- %-20s : %6.2f%%\n", "Fake rate",
             (100. * static_cast<double>(_stats.false_seeds) /
              static_cast<double>(_stats.true_seeds + _stats.false_seeds)));
    result.append(buffer);
    snprintf(
        buffer, 512, "- %-20s : %6.2f%%\n", "Recall/Efficiency",
        (100. * static_cast<double>(_stats.matched_tracks) /
         static_cast<double>(_stats.unmatched_tracks + _stats.matched_tracks)));
    result.append(buffer);

    return result;
}
}  // namespace traccc
