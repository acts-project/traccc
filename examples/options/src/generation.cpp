/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/generation.hpp"

#include "traccc/utils/particle.hpp"
#include "traccc/utils/ranges.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

generation::generation() : interface("Particle Generation Options") {

    m_desc.add_options()("gen-events",
                         po::value(&events)->default_value(events),
                         "The number of events to generate");
    m_desc.add_options()(
        "gen-nparticles",
        po::value(&gen_nparticles)->default_value(gen_nparticles),
        "The number of particles to generate per event");
    m_desc.add_options()(
        "gen-vertex-xyz-mm",
        po::value(&vertex)->value_name("X:Y:Z")->default_value(vertex),
        "Vertex [mm]");
    m_desc.add_options()("gen-vertex-xyz-std-mm",
                         po::value(&vertex_stddev)
                             ->value_name("X:Y:Z")
                             ->default_value(vertex_stddev),
                         "Standard deviation of the vertex [mm]");
    m_desc.add_options()(
        "gen-mom-gev",
        po::value(&mom_range)->value_name("MIN:MAX")->default_value(mom_range),
        "Range of momentum [GeV]");
    m_desc.add_options()(
        "gen-phi-degree",
        po::value(&phi_range)->value_name("MIN:MAX")->default_value(phi_range),
        "Range of phi [Degree]");
    m_desc.add_options()(
        "gen-eta",
        po::value(&eta_range)->value_name("MIN:MAX")->default_value(eta_range),
        "Range of eta");
    m_desc.add_options()("particle-type",
                         po::value<int>(&pdg_number)->default_value(pdg_number),
                         "PDG number for the particle type");
}

void generation::read(const po::variables_map&) {

    vertex *= detray::unit<float>::mm;
    vertex_stddev *= detray::unit<float>::mm;
    mom_range *= detray::unit<float>::GeV;
    phi_range *= detray::unit<float>::degree;
    theta_range = eta_to_theta_range(eta_range);
    ptc_type = detail::particle_from_pdg_number<traccc::scalar>(pdg_number);
}

std::ostream& generation::print_impl(std::ostream& out) const {

    out << "  Number of events to generate   : " << events << "\n"
        << "  Number of particles to generate: " << gen_nparticles << "\n"
        << "  Vertex                         : "
        << vertex / detray::unit<float>::mm << " mm\n"
        << "  Vertex standard deviation      : "
        << vertex_stddev / detray::unit<float>::mm << " mm\n"
        << "  Momentum range                 : "
        << mom_range / detray::unit<float>::GeV << " GeV\n"
        << "  Phi range                      : "
        << phi_range / detray::unit<float>::degree << " deg\n"
        << "  Eta range                      : " << eta_range << "\n"
        << "  Theta range                    : "
        << theta_range / detray::unit<float>::degree << " deg\n"
        << "  PDG Number                     : " << pdg_number;
    return out;
}

}  // namespace traccc::opts
