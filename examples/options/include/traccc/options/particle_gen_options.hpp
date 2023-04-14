/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/options.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

// Boost
#include <boost/program_options.hpp>

namespace traccc {

namespace po = boost::program_options;

template <typename scalar_t>
struct particle_gen_options {
    unsigned int gen_nparticles{1};
    Reals<scalar_t, 3> vertex{0., 0., 0.};
    Reals<scalar_t, 3> vertex_stddev{0., 0., 0.};
    Reals<scalar_t, 2> mom_range{1., 1.};
    Reals<scalar_t, 2> phi_range{0., 0.};
    Reals<scalar_t, 2> theta_range{0., 0.};

    particle_gen_options(po::options_description& desc) {
        desc.add_options()("gen-nparticles",
                           po::value<unsigned int>()->default_value(1),
                           "The number of particles to generate per event");
        desc.add_options()(
            "gen-vertex-xyz-mm",
            po::value<Reals<scalar_t, 3>>()->value_name("X:Y:Z")->default_value(
                {{0.f, 0.f, 0.f}}),
            "Vertex [mm]");
        desc.add_options()(
            "gen-vertex-xyz-std-mm",
            po::value<Reals<scalar_t, 3>>()->value_name("X:Y:Z")->default_value(
                {{0.f, 0.f, 0.f}}),
            "Standard deviation of the vertex [mm]");
        desc.add_options()("gen-mom-gev",
                           po::value<Reals<scalar_t, 2>>()
                               ->value_name("MIN:MAX")
                               ->default_value({1.f, 1.f}),
                           "Range of momentum [GeV]");
        desc.add_options()("gen-phi-degree",
                           po::value<Reals<scalar_t, 2>>()
                               ->value_name("MIN:MAX")
                               ->default_value({0.f, 0.f}),
                           "Range of phi [Degree]");
        desc.add_options()("gen-eta",
                           po::value<Reals<scalar_t, 2>>()
                               ->value_name("MIN:MAX")
                               ->default_value({0.f, 0.f}),
                           "Range of eta");
    }
    void read(const po::variables_map& vm) {
        gen_nparticles = vm["gen-nparticles"].as<unsigned int>();
        vertex = vm["gen-vertex-xyz-mm"].as<Reals<scalar_t, 3>>();
        vertex_stddev = vm["gen-vertex-xyz-std-mm"].as<Reals<scalar_t, 3>>();
        mom_range = vm["gen-mom-gev"].as<Reals<scalar_t, 2>>();
        const auto phi_range_degree =
            vm["gen-phi-degree"].as<Reals<scalar_t, 2>>();
        const auto eta_range = vm["gen-eta"].as<Reals<scalar_t, 2>>();
        // @TODO: remove the conversion here...
        // @NOTE: I put eta_range[0] into theta_range[1] and eta_range[1] into
        // theta_range[0] on purpose because theta(minEta) > theta(maxEta)
        theta_range = {2 * std::atan(std::exp(-eta_range[1])),
                       2 * std::atan(std::exp(-eta_range[0]))};
        phi_range = {phi_range_degree[0] * detray::unit<scalar_t>::degree,
                     phi_range_degree[1] * detray::unit<scalar_t>::degree};
    }
};

}  // namespace traccc