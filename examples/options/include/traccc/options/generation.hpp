/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/details/interface.hpp"
#include "traccc/options/details/value_array.hpp"

// System include(s).
#include <cstddef>

namespace traccc::opts {

/// Configuration for particle / event generation
class generation : public interface {

    public:
    /// @name Configurable options
    /// @{

    /// The number of events to generate
    std::size_t events = 1;

    /// The number of particles to generate per event
    unsigned int gen_nparticles{1u};
    /// Vertex position [mm]
    opts::value_array<float, 3> vertex{0.f, 0.f, 0.f};
    /// Standard deviation of the vertex position [mm]
    opts::value_array<float, 3> vertex_stddev{0.f, 0.f, 0.f};
    /// Range of momentum [GeV]
    opts::value_array<float, 2> mom_range{1.f, 1.f};
    /// Range of phi [rad]
    opts::value_array<float, 2> phi_range{-180.f, 180.f};
    /// Range of eta
    opts::value_array<float, 2> eta_range{-2.f, 2.f};
    /// Charge of particles
    float charge{-1.f};

    /// @}

    /// @name Derived options
    /// @{

    /// Range of theta [rad]
    opts::value_array<float, 2> theta_range{0.f, 0.f};

    /// @}

    /// Constructor
    generation();

    /// Read/process the command line options
    ///
    /// @param vm The command line options to interpret/read
    ///
    void read(const boost::program_options::variables_map& vm) override;

    private:
    /// Print the specific options of this class
    std::ostream& print_impl(std::ostream& out) const override;

};  // struct generation

}  // namespace traccc::opts
