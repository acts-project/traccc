/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/options.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <iosfwd>

namespace traccc {

/// Configuration for particle generation
struct particle_gen_options {

    /// The number of particles to generate per event
    unsigned int gen_nparticles{1u};
    /// Vertex position [mm]
    Reals<float, 3> vertex{0., 0., 0.};
    /// Standard deviation of the vertex position [mm]
    Reals<float, 3> vertex_stddev{0., 0., 0.};
    /// Range of momentum [GeV]
    Reals<float, 2> mom_range{1., 1.};
    /// Range of phi [rad]
    Reals<float, 2> phi_range{0., 0.};
    /// Range of theta [rad]
    Reals<float, 2> theta_range{0., 0.};
    /// Charge of particles
    float charge{-1.f};

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    particle_gen_options(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

};  // struct particle_gen_options

/// Printout helper for @c traccc::particle_gen_options
std::ostream& operator<<(std::ostream& out, const particle_gen_options& opt);

}  // namespace traccc