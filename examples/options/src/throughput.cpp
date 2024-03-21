/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/options/throughput.hpp"

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Throughput Measurement Options";

throughput::throughput(po::options_description& desc) : m_desc{description} {

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
    desc.add(m_desc);
}

void throughput::read(const po::variables_map&) {}

std::ostream& operator<<(std::ostream& out, const throughput& opt) {

    out << ">>> " << description << " <<<\n"
        << "  Cold run event(s)          : " << opt.cold_run_events << "\n"
        << "  Processed event(s)         : " << opt.processed_events << "\n"
        << "  Log file                   : " << opt.log_file;
    return out;
}

}  // namespace traccc::opts
