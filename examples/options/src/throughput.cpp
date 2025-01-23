/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/options/throughput.hpp"

#include "traccc/examples/utils/printable.hpp"

// System include(s).
#include <iostream>

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
    m_desc.add_options()(
        "log-file", po::value(&log_file),
        "File where result logs will be printed (in append mode).");
}

std::unique_ptr<configuration_printable> throughput::as_printable() const {
    std::unique_ptr<configuration_printable> cat =
        std::make_unique<configuration_category>(
            "Throughput measurement options");

    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Cold run events", std::to_string(cold_run_events)));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>(
            "Processed events", std::to_string(processed_events)));
    dynamic_cast<configuration_category &>(*cat).add_child(
        std::make_unique<configuration_kv_pair>("Log file", log_file));

    return cat;
}

}  // namespace traccc::opts
