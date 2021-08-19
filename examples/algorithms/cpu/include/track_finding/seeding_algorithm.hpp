/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "seeding/seed_finding.hpp"
#include "seeding/spacepoint_grouping.hpp"

namespace traccc {

class seeding_algorithm
    : public algorithm<
          const host_spacepoint_container&,
          std::pair<host_internal_spacepoint_container, host_seed_container> > {
    public:
    seeding_algorithm(vecmem::memory_resource* mr = nullptr) : m_mr(mr) {
        // Seed finder config
        // silicon detector max
        m_config.rMax = 160.;
        m_config.deltaRMin = 5.;
        m_config.deltaRMax = 160.;
        m_config.collisionRegionMin = -250.;
        m_config.collisionRegionMax = 250.;
        // m_config.zMin = -2800.; // this value introduces redundant bins
        // without any spacepoints m_config.zMax = 2800.;
        m_config.zMin = -1186.;
        m_config.zMax = 1186.;
        m_config.maxSeedsPerSpM = 5;
        // 2.7 eta
        m_config.cotThetaMax = 7.40627;
        m_config.sigmaScattering = 1.00000;

        m_config.minPt = 500.;
        m_config.bFieldInZ = 0.00199724;

        m_config.beamPos = {-.5, -.5};
        m_config.impactMax = 10.;

        m_config.highland = 13.6 * std::sqrt(m_config.radLengthPerSeed) *
                            (1 + 0.038 * std::log(m_config.radLengthPerSeed));
        float maxScatteringAngle = m_config.highland / m_config.minPt;
        m_config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;
        // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV
        // and millimeter
        // TODO: change using ACTS units
        m_config.pTPerHelixRadius = 300. * m_config.bFieldInZ;
        m_config.minHelixDiameter2 =
            std::pow(m_config.minPt * 2 / m_config.pTPerHelixRadius, 2);
        m_config.pT2perRadius =
            std::pow(m_config.highland / m_config.pTPerHelixRadius, 2);

        m_grid_config.bFieldInZ = m_config.bFieldInZ;
        m_grid_config.minPt = m_config.minPt;
        m_grid_config.rMax = m_config.rMax;
        m_grid_config.zMax = m_config.zMax;
        m_grid_config.zMin = m_config.zMin;
        m_grid_config.deltaRMax = m_config.deltaRMax;
        m_grid_config.cotThetaMax = m_config.cotThetaMax;
    }

    output_type operator()(
        const input_type& i) const override {
        output_type o;
        this->operator()(i, o);
        return o;
    }

    void operator()(const input_type& spacepoints_per_event,
                    output_type& o) const override {
        // output containers
        auto& internal_sp_per_event = o.first;
        auto& seeds = o.second;

        // spacepoint grouping
	traccc::spacepoint_grouping sg(m_config, m_grid_config, m_mr);
        internal_sp_per_event = sg(spacepoints_per_event);
	
        // seed finding
	traccc::seed_finding sf(m_config);
        seeds = sf(internal_sp_per_event);
    }

private:
    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    vecmem::memory_resource* m_mr;
};

}  // namespace traccc
