/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/options.hpp"

// Boost
#include <boost/program_options.hpp>

namespace traccc {

namespace po = boost::program_options;

struct finding_input_config {
    Reals<unsigned int, 2> track_candidates_range;
    bool check_performance;

    finding_input_config(po::options_description& desc);
    void read(const po::variables_map& vm);
};

}  // namespace traccc