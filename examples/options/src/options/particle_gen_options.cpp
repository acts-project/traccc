/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/particle_gen_options.hpp"

#include "traccc/utils/ranges.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

namespace traccc {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Type alias for the eta range argument
using eta_range_type = opts::value_array<float, 2>;
/// Name of the eta range option
static const char* eta_range_option = "gen-eta";

particle_gen_options::particle_gen_options(po::options_description& desc) {

    desc.add_options()(
        "gen-nparticles",
        po::value(&gen_nparticles)->default_value(gen_nparticles),
        "The number of particles to generate per event");
    desc.add_options()(
        "gen-vertex-xyz-mm",
        po::value(&vertex)->value_name("X:Y:Z")->default_value(vertex),
        "Vertex [mm]");
    desc.add_options()("gen-vertex-xyz-std-mm",
                       po::value(&vertex_stddev)
                           ->value_name("X:Y:Z")
                           ->default_value(vertex_stddev),
                       "Standard deviation of the vertex [mm]");
    desc.add_options()(
        "gen-mom-gev",
        po::value(&mom_range)->value_name("MIN:MAX")->default_value(mom_range),
        "Range of momentum [GeV]");
    desc.add_options()(
        "gen-phi-degree",
        po::value(&phi_range)->value_name("MIN:MAX")->default_value(phi_range),
        "Range of phi [Degree]");
    desc.add_options()(
        eta_range_option,
        po::value<eta_range_type>()->value_name("MIN:MAX")->default_value(
            {0.f, 0.f}),
        "Range of eta");
    desc.add_options()("charge", po::value(&charge)->default_value(charge),
                       "Charge of particles");
}

void particle_gen_options::read(const po::variables_map& vm) {

    phi_range[0] *= detray::unit<float>::degree;
    phi_range[1] *= detray::unit<float>::degree;
    theta_range = eta_to_theta_range(vm[eta_range_option].as<eta_range_type>());
}

std::ostream& operator<<(std::ostream& out, const particle_gen_options& opt) {

    out << ">>> Particle generation options: <<<\n"
        << "  Number of particles to generate : " << opt.gen_nparticles << "\n"
        << "  Vertex                          : " << opt.vertex << " mm\n"
        << "  Vertex standard deviation       : " << opt.vertex_stddev
        << " mm\n"
        << "  Momentum range                  : " << opt.mom_range << " GeV\n"
        << "  Phi range                       : " << opt.phi_range << " rad\n"
        << "  Theta range                     : " << opt.theta_range << " rad\n"
        << "  Charge                          : " << opt.charge << "\n";
    return out;
}

}  // namespace traccc
