/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <chrono>
#include <iostream>

#include "../../../device/sycl/include/sycl/seeding/seed_finding.hpp"
#include "seeding/spacepoint_grouping.hpp"

namespace traccc {
namespace sycl {

class seeding_algorithm {
    public:
    using input_type = host_spacepoint_container&;
    using output_type =
        std::pair<host_internal_spacepoint_container, host_seed_container>;

    seeding_algorithm(vecmem::memory_resource* mr = nullptr, ::sycl::queue* q = nullptr) : m_mr(mr), m_q(q) {

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

        // multiplet estimator
        m_estimator.m_cfg.safety_factor = 2.0;
        m_estimator.m_cfg.safety_adder = 10;
        // m_estimator.m_cfg.safety_factor = 10.0;
        // m_estimator.m_cfg.safety_adder = 50000;
        m_estimator.m_cfg.par_for_mb_doublets = {1, 28.77, 0.4221};
        m_estimator.m_cfg.par_for_mt_doublets = {1, 19.73, 0.232};
        m_estimator.m_cfg.par_for_triplets = {1, 0, 0.02149};
        m_estimator.m_cfg.par_for_seeds = {0, 0.3431};

        sg = std::make_shared<traccc::spacepoint_grouping>(
            traccc::spacepoint_grouping(m_config, m_grid_config, m_mr));
        sf = std::make_shared<traccc::sycl::seed_finding>(
            traccc::sycl::seed_finding(m_config, sg->get_spgrid(), m_estimator,
                                       m_mr, m_q));
    }

    output_type operator()(input_type& i) {
        output_type o;
        this->operator()(i, o);
        return o;
    }

    void operator()(input_type& spacepoints_per_event, output_type& o) {
        // spacepoint grouping
        auto internal_sp_per_event = sg->operator()(spacepoints_per_event);

        // seed finding
        auto seeds = sf->operator()(internal_sp_per_event);

        // output container
        o.first = std::move(internal_sp_per_event);
        o.second = std::move(seeds);
    }

    private:
    seedfinder_config m_config;
    spacepoint_grid_config m_grid_config;
    multiplet_estimator m_estimator;
    std::shared_ptr<traccc::spacepoint_grouping> sg;
    std::shared_ptr<traccc::sycl::seed_finding> sf;
    vecmem::memory_resource* m_mr;
    ::sycl::queue* m_q;
};

}  // namespace sycl
}  // namespace traccc
