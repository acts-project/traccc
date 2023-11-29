/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/options.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

// Boost
#include <boost/program_options.hpp>

namespace traccc {

namespace po = boost::program_options;

template <typename scalar_t>
struct telescope_detector_options {
    bool empty_material;
    unsigned int n_planes;
    scalar_t thickness;
    scalar_t spacing;
    scalar_t smearing;
    scalar_t half_length;

    telescope_detector_options(po::options_description& desc) {
        desc.add_options()("empty_material",
                           po::value<bool>()->default_value(false),
                           "Build detector without materials");
        desc.add_options()("n_planes",
                           po::value<unsigned int>()->default_value(9),
                           "Number of planes");
        desc.add_options()("thickness",
                           po::value<scalar_t>()->default_value(0.5f),
                           "Slab thickness in [mm]");
        desc.add_options()("spacing",
                           po::value<scalar_t>()->default_value(20.f),
                           "Space between planes in [mm]");
        desc.add_options()("smearing",
                           po::value<scalar_t>()->default_value(50.f),
                           "Measurement smearing in [um]");
        desc.add_options()("half_length",
                           po::value<scalar_t>()->default_value(1000000.f),
                           "Half length of plane [mm]");
    }

    void read(const po::variables_map& vm) {
        empty_material = vm["empty_material"].as<bool>();
        n_planes = vm["n_planes"].as<unsigned int>();
        thickness = vm["thickness"].as<scalar_t>() * detray::unit<scalar_t>::mm;
        spacing = vm["spacing"].as<scalar_t>() * detray::unit<scalar_t>::mm;
        smearing = vm["smearing"].as<scalar_t>() * detray::unit<scalar_t>::um;
        half_length =
            vm["half_length"].as<scalar_t>() * detray::unit<scalar_t>::mm;
    }
};

}  // namespace traccc