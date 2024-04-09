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
#include <cstddef>
#include <string>

namespace traccc::opts {

/// Options for the input data that a given application would use
class input_data : public interface {

    public:
    /// @name Options
    /// @{

    /// The data format of the input files
    traccc::data_format format = data_format::csv;
    /// Directory of the input files
    std::string directory = "tml_full/ttbar_mu20/";
    /// The number of events to process
    std::size_t events = 1;
    /// The number of events to skip
    std::size_t skip = 0;

    /// @}

    /// Constructor
    input_data();

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm) override;

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // class input_data

}  // namespace traccc::opts
