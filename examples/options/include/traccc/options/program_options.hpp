/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/options/details/interface.hpp"

#include <boost/program_options.hpp>

#include <functional>
#include <string_view>
#include <vector>

namespace traccc::opts {

/// Top-level propgram options for an executable
class program_options {

    public:
    /// Constructor
    program_options(
        std::string_view description,
        const std::vector<std::reference_wrapper<interface> >& options,
        int argc, char* argv[]);

    private:
    /// Description of all program options
    boost::program_options::options_description m_desc;

};  // class program_options

}  // namespace traccc::opts
