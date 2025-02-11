/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/options/throughput.hpp"

#include "traccc/examples/utils/printable.hpp"

// System include(s).
#include <format>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

throughput::throughput() : interface("Throughput Measurement Options") {

    m_desc.add_options()(
        "processed-events",
        po::value(&processed_events)->default_value(processed_events),
        "Number of events to process");
    m_desc.add_options()(
        "cold-run-events",
        po::value(&cold_run_events)->default_value(cold_run_events),
        "Number of events to run 'cold'");
    m_desc.add_options()("deterministic",
                         po::bool_switch(&deterministic_event_order)
                             ->default_value(deterministic_event_order),
                         "Process events in deterministic order");
    m_desc.add_options()("random-seed",
                         po::value(&random_seed)->default_value(random_seed),
                         "Seed for event randomization (0 to use time)");
    m_desc.add_options()(
        "log-file", po::value(&log_file),
        "File where result logs will be printed (in append mode).");
}

std::unique_ptr<configuration_printable> throughput::as_printable() const {
    auto cat = std::make_unique<configuration_category>(m_description);

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Cold run events", std::to_string(cold_run_events)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Processed events", std::to_string(processed_events)));
    cat->add_child(
        std::make_unique<configuration_kv_pair>("Log file", log_file));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Deterministic ordering",
        std::format("{}", deterministic_event_order)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "Random seed",
        random_seed == 0 ? "time-based" : std::to_string(random_seed)));

    return cat;
}

}  // namespace traccc::opts
