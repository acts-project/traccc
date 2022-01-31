/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <chrono>
#include <iostream>

#include "../../../device/sycl/include/sycl/seeding/seed_finding.hpp"
#include "../../../device/sycl/include/sycl/seeding/spacepoint_binning.hpp"

namespace traccc {
namespace sycl {

class seeding_algorithm
    : public algorithm<host_seed_container(host_spacepoint_container&&)> {

    public:
    seeding_algorithm(vecmem::memory_resource& mr, ::sycl::queue* q = nullptr)
        : m_mr(mr), m_q(q) {

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

        sb = std::make_shared<traccc::sycl::spacepoint_binning>(
            traccc::sycl::spacepoint_binning(m_config, m_grid_config, mr, q));
        sf = std::make_shared<traccc::sycl::seed_finding>(
            traccc::sycl::seed_finding(m_config, sb->nbins(), mr, q));
    }

    output_type operator()(
        host_spacepoint_container&& spacepoints) const override {
        output_type seeds(1, &m_mr.get());
        auto internal_sp_g2 = sb->operator()(std::move(spacepoints));
        seeds =
            sf->operator()(std::move(spacepoints), std::move(internal_sp_g2));
        return seeds;
    }

    private:
    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
    ::sycl::queue* m_q;
    std::shared_ptr<traccc::sycl::spacepoint_binning> sb;
    std::shared_ptr<traccc::sycl::seed_finding> sf;
};

}  // namespace sycl
}  // namespace traccc
