/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/options/generation.hpp"

#include "traccc/utils/ranges.hpp"

// Detray include(s).
#include "detray/definitions/units.hpp"

namespace traccc::opts {

/// Convenience namespace shorthand
namespace po = boost::program_options;

/// Description of this option group
static const char* description = "Particle Generation Options";

generation::generation(po::options_description& desc) : m_desc{description} {

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
    m_desc.add_options()("charge", po::value(&charge)->default_value(charge),
                         "Charge of particles");
    desc.add(m_desc);
}

void generation::read(const po::variables_map&) {

    phi_range[0] *= detray::unit<float>::degree;
    phi_range[1] *= detray::unit<float>::degree;
    theta_range = eta_to_theta_range(eta_range);
}

std::ostream& operator<<(std::ostream& out, const generation& opt) {

    out << ">>> " << description << " <<<\n"
        << "  Number of events to generate    : " << opt.events << "\n"
        << "  Number of particles to generate : " << opt.gen_nparticles << "\n"
        << "  Vertex                          : " << opt.vertex << " mm\n"
        << "  Vertex standard deviation       : " << opt.vertex_stddev
        << " mm\n"
        << "  Momentum range                  : " << opt.mom_range << " GeV\n"
        << "  Phi range                       : " << opt.phi_range << " rad\n"
        << "  Eta range                       : " << opt.eta_range << "\n"
        << "  Theta range                     : " << opt.theta_range << " rad\n"
        << "  Charge                          : " << opt.charge;
    return out;
}

}  // namespace traccc::opts
