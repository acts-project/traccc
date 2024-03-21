/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/value_array.hpp"

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <cstddef>
#include <iosfwd>

namespace traccc::opts {

/// Configuration for particle / event generation
class generation {

    public:
    /// @name Options
    /// @{

    /// The number of events to generate
    std::size_t events = 1;

    /// The number of particles to generate per event
    unsigned int gen_nparticles{1u};
    /// Vertex position [mm]
    opts::value_array<float, 3> vertex{0., 0., 0.};
    /// Standard deviation of the vertex position [mm]
    opts::value_array<float, 3> vertex_stddev{0., 0., 0.};
    /// Range of momentum [GeV]
    opts::value_array<float, 2> mom_range{1., 1.};
    /// Range of phi [rad]
    opts::value_array<float, 2> phi_range{0., 0.};
    /// Range of eta
    opts::value_array<float, 2> eta_range{0., 0.};
    /// Range of theta [rad]
    opts::value_array<float, 2> theta_range{0., 0.};
    /// Charge of particles
    float charge{-1.f};

    /// @}

    /// Constructor on top of a common @c program_options object
    ///
    /// @param desc The program options to add to
    ///
    generation(boost::program_options::options_description& desc);

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm);

    private:
    /// The program options group
    boost::program_options::options_description m_desc;

};  // struct generation

/// Printout helper for @c traccc::opts::generation
std::ostream& operator<<(std::ostream& out, const generation& opt);

}  // namespace traccc::opts
