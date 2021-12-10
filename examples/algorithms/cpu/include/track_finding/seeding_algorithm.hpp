/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <chrono>
#include <iostream>

#include "seeding/seed_finding.hpp"
#include "seeding/spacepoint_binning.hpp"

namespace traccc {

class seeding_algorithm
    : public algorithm<host_seed_container(const host_spacepoint_container&)> {
    public:
    seeding_algorithm(vecmem::memory_resource& mr) : m_mr(mr) {

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

        sb = std::make_shared<traccc::spacepoint_binning>(
            traccc::spacepoint_binning(m_config, m_grid_config, mr));
        sf = std::make_shared<traccc::seed_finding>(
            traccc::seed_finding(m_config));
    }

    output_type operator()(const host_spacepoint_container& i) const override {
        output_type o;
        this->operator()(i, o);
        return o;
    }

    void operator()(const host_spacepoint_container& spacepoints_per_event,
                    output_type& seeds) const {
        auto internal_sp_g2 = sb->operator()(spacepoints_per_event);
        seeds = sf->operator()(internal_sp_g2);
    }

    seedfinder_config get_seedfinder_config() { return m_config; }

    private:
    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    std::shared_ptr<traccc::spacepoint_binning> sb;
    std::shared_ptr<traccc::seed_finding> sf;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc
