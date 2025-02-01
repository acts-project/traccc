/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"

// System include(s).
#include <cstddef>
#include <string>

namespace traccc::opts {

/// Command line options used in the throughput tests
class throughput : public interface {

    public:
    /// @name Options
    /// @{

    /// The number of events to process during the job
    std::size_t processed_events = 100;
    /// The number of events to run "cold", i.e. run without accounting for
    /// them in the performance measurements
    std::size_t cold_run_events = 10;

    /// Enable or disable the randomization of event processing
    bool deterministic_event_order = false;
    /// Set the random event processing seed
    unsigned int random_seed = 0;

    /// Output log file
    std::string log_file;

    /// @}

    /// Constructor
    throughput();

    std::unique_ptr<configuration_printable> as_printable() const override;
};  // class throughput

}  // namespace traccc::opts
