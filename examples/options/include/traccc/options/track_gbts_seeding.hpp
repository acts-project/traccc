/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"

// System include(s).
#include <cstddef>

namespace traccc::opts {

/// Option(s) for multi-threaded code execution
class track_gbts_seeding : public interface {

    public:
    /// @name Options
    /// @{

    bool useGBTS = false;
    std::string config_dir = "DEFAULT";
	
    /// @}
	// config info from file
	std::vector<std::pair<uint64_t, short>> barcodeBinning;
	std::vector<std::pair<int, std::vector<int>>> binTables;
	traccc::device::gbts_layerInfo layerInfo;

    /// Constructor
    track_gbts_seeding();

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm) override;

    std::unique_ptr<configuration_printable> as_printable() const override;
};

}  // namespace traccc::opts
