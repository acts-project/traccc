/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

namespace traccc::opts {

/// Command line options used in the telescope detector tests
class telescope_detector : public interface {

    public:
    /// @name Options
    /// @{

    /// Build detector without materials
    bool empty_material = false;
    /// Number of planes
    unsigned int n_planes = 9;
    /// Slab thickness in [mm]
    float thickness = 0.5f;
    /// Space between planes in [mm]
    float spacing = 20.f;
    /// Measurement smearing in [um]
    float smearing = 50.f;
    /// Half length of plane [mm]
    float half_length = 1000000.f;

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    telescope_detector(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm) override;

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // ckass telescope_detector

}  // namespace traccc::opts
