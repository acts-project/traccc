/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>

namespace traccc {

/// Command line options used in the telescope detector tests
struct telescope_detector_options {

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

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    telescope_detector_options(
        boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct telescope_detector_options

/// Printout helper for @c traccc::telescope_detector_options
std::ostream& operator<<(std::ostream& out,
                         const telescope_detector_options& opt);

}  // namespace traccc
