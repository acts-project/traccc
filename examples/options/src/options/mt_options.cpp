/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/options/mt_options.hpp"

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc {

mt_options::mt_options(boost::program_options::options_description& desc) {

    desc.add_options()(
        "threads",
        boost::program_options::value<std::size_t>()->default_value(1),
        "The number of CPU threads to use");
}

void mt_options::read(const boost::program_options::variables_map& vm) {

    threads = vm["threads"].as<std::size_t>();
    if (threads == 0) {
        throw std::invalid_argument{"Must use threads>0"};
    }
}

std::ostream& operator<<(std::ostream& out, const mt_options& opt) {

    out << ">>> Multi-threading options <<<\n"
        << "CPU threads: " << opt.threads;
    return out;
}

}  // namespace traccc
