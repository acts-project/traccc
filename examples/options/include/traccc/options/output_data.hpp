/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/io/data_format.hpp"
#include "traccc/options/details/interface.hpp"

// System include(s).
#include <string>

namespace traccc::opts {

/// Options for the output data that a given application would produce
class output_data : public interface {

    public:
    /// @name Options
    /// @{

    /// The data format of the input files
    traccc::data_format format = data_format::csv;
    /// Directory of the input files
    std::string directory = "testing/";

    /// @}

    /// Constructor
    output_data();

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm) override;

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // class output_data

}  // namespace traccc::opts
