/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/track_gbts_seeding.hpp"

#include "traccc/examples/utils/printable.hpp"

// System include(s).
#include <stdexcept>
#include <iostream>
namespace traccc::opts {

track_gbts_seeding::track_gbts_seeding() : interface("GBTS Options") {
    m_desc.add_options()(
        "useGBTS",
        boost::program_options::value(&useGBTS)->default_value(useGBTS),
        "use gbts algorithm");
    
    m_desc.add_options()(
        "gbts_config_dir",
        boost::program_options::value(&config_dir)->default_value(config_dir),
        "directory for gbts config files");
}

void track_gbts_seeding::read(const boost::program_options::variables_map &) {
	// where make config
}

std::unique_ptr<configuration_printable> track_gbts_seeding::as_printable() const {
    auto cat = std::make_unique<configuration_category>(m_description);

    cat->add_child(std::make_unique<configuration_kv_pair>(
        "using gbts algorithm ", std::to_string(useGBTS)));
    cat->add_child(std::make_unique<configuration_kv_pair>(
        "gbts config directory ", config_dir));

    return cat;
}

}  // namespace traccc::opts
