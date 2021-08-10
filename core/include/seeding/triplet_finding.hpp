/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/doublet.hpp>
#include <seeding/detail/triplet.hpp>
#include <seeding/triplet_finding_helper.hpp>

namespace traccc {

/// Triplet finding to search the compatible combintations of two doublets which
/// share same middle spacepoint
struct triplet_finding {
    /// Constructor for the triplet finding
    ///
    /// @param seedfinder_config is the configuration parameters
    /// @param isp_container is the internal spacepoint container
    triplet_finding(seedfinder_config& config) : m_config(config) {}

    /// Callable operator for triplet finding per middle-bottom doublet
    ///
    /// @param mid_bot is the current middle-bottom doublets
    /// @param lb is transformed coordinate of mid_bot
    /// @param doublets_mid_top is the vector of middle-top doublets which share
    /// same middle spacepoint with current middle-bottom doublet
    /// @param lin_circles_mid_top is transformed coordinates of
    /// doublets_mid_top
    ///
    /// @return a vector of triplets
    host_triplet_collection operator()(
        const host_internal_spacepoint_container& isp_container,
        const doublet& mid_bot, const lin_circle& lb,
        const host_doublet_collection& doublets_mid_top,
        const host_lin_circle_collection& lin_circles_mid_top) {
        host_triplet_collection triplets;
        this->operator()(isp_container, mid_bot, lb, doublets_mid_top,
                         lin_circles_mid_top, triplets);
        return triplets;
    }

    /// Callable operator for triplet finding per middle-bottom doublet
    ///
    /// @param mid_bot is the current middle-bottom doublets
    /// @param lb is transformed coordinate of mid_bot
    /// @param doublets_mid_top is the vector of middle-top doublets which share
    /// same middle spacepoint with current middle-bottom doublet
    /// @param lin_circles_mid_top is transformed coordinates of
    /// doublets_mid_top
    ///
    /// void interface
    ///
    /// @return a vector of triplets
    void operator()(const host_internal_spacepoint_container& isp_container,
                    const doublet& mid_bot, const lin_circle& lb,
                    const host_doublet_collection& doublets_mid_top,
                    const host_lin_circle_collection& lin_circles_mid_top,
                    host_triplet_collection& triplets) {
        auto& spM_idx = mid_bot.sp1;
        auto& spM = isp_container.items[spM_idx.bin_idx][spM_idx.sp_idx];

        scalar iSinTheta2 = 1 + lb.cotTheta() * lb.cotTheta();
        scalar scatteringInRegion2 = m_config.maxScatteringAngle2 * iSinTheta2;
        scatteringInRegion2 *=
            m_config.sigmaScattering * m_config.sigmaScattering;
        scalar curvature, impact_parameter;

        for (size_t i = 0; i < doublets_mid_top.size(); ++i) {
            auto& mid_top = doublets_mid_top[i];
            auto& lt = lin_circles_mid_top[i];

            if (!triplet_finding_helper::isCompatible(
                    spM, lb, lt, m_config, iSinTheta2, scatteringInRegion2,
                    curvature, impact_parameter)) {
                continue;
            }

            triplets.push_back(
                {mid_bot.sp2,  // bottom
                 mid_bot.sp1,  // middle
                 mid_top.sp2,  // top
                 curvature,    // curvature
                 -impact_parameter * m_filter_config.impactWeightFactor,
                 lb.Zo()});
        }

        for (size_t i = 0; i < triplets.size(); ++i) {
            auto& current_triplet = triplets[i];
            auto& spT_idx = current_triplet.sp3;
            auto& current_spT =
                isp_container.items[spT_idx.bin_idx][spT_idx.sp_idx];
            const auto& currentTop_r = current_spT.radius();

            // if two compatible seeds with high distance in r are found,
            // compatible seeds span 5 layers
            // -> very good seed
            std::vector<float> compatibleSeedR;
            float lowerLimitCurv = current_triplet.curvature -
                                   m_filter_config.deltaInvHelixDiameter;
            float upperLimitCurv = current_triplet.curvature +
                                   m_filter_config.deltaInvHelixDiameter;

            for (size_t j = 0; j < triplets.size(); ++j) {
                if (i == j) {
                    continue;
                }

                auto& other_triplet = triplets[j];
                auto& other_spT_idx = other_triplet.sp3;
                auto& other_spT =
                    isp_container
                        .items[other_spT_idx.bin_idx][other_spT_idx.sp_idx];

                // compared top SP should have at least deltaRMin distance
                const auto& otherTop_r = other_spT.radius();
                float deltaR = currentTop_r - otherTop_r;
                if (std::abs(deltaR) < m_filter_config.deltaRMin) {
                    continue;
                }

                // curvature difference within limits?
                // TODO: how much slower than sorting all vectors by curvature
                // and breaking out of loop? i.e. is vector size large (e.g. in
                // jets?)
                if (other_triplet.curvature < lowerLimitCurv) {
                    continue;
                }
                if (other_triplet.curvature > upperLimitCurv) {
                    continue;
                }

                bool newCompSeed = true;
                for (float previousDiameter : compatibleSeedR) {
                    // original ATLAS code uses higher min distance for 2nd
                    // found compatible seed (20mm instead of 5mm) add new
                    // compatible seed only if distance larger than rmin to all
                    // other compatible seeds
                    if (std::abs(previousDiameter - otherTop_r) <
                        m_filter_config.deltaRMin) {
                        newCompSeed = false;
                        break;
                    }
                }

                if (newCompSeed) {
                    compatibleSeedR.push_back(otherTop_r);
                    current_triplet.weight += m_filter_config.compatSeedWeight;
                }

                if (compatibleSeedR.size() >= m_filter_config.compatSeedLimit) {
                    break;
                }
            }
        }
    }

    private:
    seedfinder_config m_config;
    seedfilter_config m_filter_config;
};

}  // namespace traccc
