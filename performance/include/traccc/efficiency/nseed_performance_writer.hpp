/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <traccc/edm/alt_measurement.hpp>
#include <traccc/edm/nseed.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/efficiency/track_filter.hpp>
#include <traccc/efficiency/track_matcher.hpp>
#include <traccc/io/event_map.hpp>

namespace traccc {
class nseed_performance_writer {
    public:
    struct statistics {
        std::size_t true_seeds = 0, false_seeds = 0;
        std::size_t matched_tracks = 0, unmatched_tracks = 0;
    };

    nseed_performance_writer(const std::string& p,
                             std::unique_ptr<track_filter>&& f,
                             std::unique_ptr<track_matcher>&& m);

    void initialize();
    void finalize();

    template <typename SeedIt, typename SpIt>
    void register_event(std::size_t ev, const SeedIt sb, const SeedIt se,
                        const SpIt pb, const event_map& em) {
        std::size_t seed_id = 0;

        std::multiset<std::size_t> matched_tracks;

        for (SeedIt s = sb; s != se; ++s) {
            std::vector<std::vector<uint64_t>> particle_ids;

            std::transform(
                s->cbegin(), s->cend(), std::back_inserter(particle_ids),
                [pb,
                 &em](const spacepoint_collection_types::host::size_type& l) {
                    traccc::alt_measurement meas = (pb + l)->meas;

                    const auto& ptcs = em.meas_ptc_map.find(meas)->second;

                    std::vector<uint64_t> ptc_ids;

                    for (auto const& [ptc, _] : ptcs) {
                        ptc_ids.push_back(ptc.particle_id);
                    }

                    return ptc_ids;
                });

            std::optional<uint64_t> pid = _matcher->operator()(particle_ids);

            if (pid) {
                _stats.true_seeds++;

                matched_tracks.insert(*pid);
            } else {
                _stats.false_seeds++;
            }

            write_seed_row(ev, seed_id++, s->size(), pid);
        }

        for (const auto& [_, ptc] : em.ptc_map) {
            bool pass = _filter->operator()(ptc);

            if (pass) {
                if (matched_tracks.count(ptc.particle_id) > 0) {
                    _stats.matched_tracks++;
                } else {
                    _stats.unmatched_tracks++;
                }
            }

            const scalar eta = getter::eta(ptc.mom);
            const scalar phi = getter::phi(ptc.mom);
            const scalar pT = getter::perp(ptc.mom);

            write_track_row(ev, ptc.particle_id, pass, ptc.charge, eta, phi,
                            pT);
        }
    }

    std::string generate_report_str() const;

    private:
    static constexpr std::string_view sep = ",";

    void write_seed_header();
    void write_seed_row(std::size_t, std::size_t, std::size_t,
                        std::optional<std::size_t>);

    void write_track_header();
    void write_track_row(std::size_t, std::size_t, bool, int, scalar, scalar,
                         scalar);

    std::string _prefix;

    std::unique_ptr<track_filter> _filter;
    std::unique_ptr<track_matcher> _matcher;

    std::ofstream output_seed_file, output_track_file;

    statistics _stats;
};
}  // namespace traccc
