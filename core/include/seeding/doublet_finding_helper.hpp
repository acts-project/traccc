/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <edm/internal_spacepoint.hpp>
#include <seeding/detail/seeding_config.hpp>

namespace traccc {

// helper functions used for both cpu and gpu
struct doublet_finding_helper {
    /// Check if two spacepoints form doublets
    ///
    /// @param sp1 is middle spacepoint
    /// @param sp2 is bottom or top spacepoint
    /// @param config is configuration parameter
    /// @param bottom is whether it is for middle-bottom or middle-top doublet
    ///
    /// @return boolean value for compatibility
    static __CUDA_HOST_DEVICE__ bool isCompatible(
        const internal_spacepoint<spacepoint>& sp1,
        const internal_spacepoint<spacepoint>& sp2,
        const seedfinder_config& config, bool bottom);

    /// Do the conformal transformation on doublet's coordinate
    ///
    /// @param sp1 is middle spacepoint
    /// @param sp2 is bottom or top spacepoint
    /// @param bottom is whether it is for middle-bottom or middle-top doublet
    ///
    /// @reutrn lin_circle which contains the transformed coordinate information
    static __CUDA_HOST_DEVICE__ lin_circle transform_coordinates(
        const internal_spacepoint<spacepoint>& sp1,
        const internal_spacepoint<spacepoint>& sp2, bool bottom);
};

bool doublet_finding_helper::isCompatible(
    const internal_spacepoint<spacepoint>& sp1,
    const internal_spacepoint<spacepoint>& sp2, const seedfinder_config& config,
    bool bottom) {
    if (bottom) {
        float deltaR = sp1.radius() - sp2.radius();
        // if r-distance is too big, try next SP in bin
        if (deltaR > config.deltaRMax) {
            return false;
        }
        // if r-distance is too small, continue because bins are NOT r-sorted
        if (deltaR < config.deltaRMin) {
            return false;
        }
        float cotTheta = (sp1.z() - sp2.z()) / deltaR;
        if (std::fabs(cotTheta) > config.cotThetaMax) {
            return false;
        }
        float zOrigin = sp1.z() - sp1.radius() * cotTheta;
        if (zOrigin < config.collisionRegionMin ||
            zOrigin > config.collisionRegionMax) {
            return false;
        }
    } else if (!bottom) {
        float deltaR = sp2.radius() - sp1.radius();
        if (deltaR > config.deltaRMax) {
            return false;
        }
        // if r-distance is too small, continue because bins are NOT r-sorted
        if (deltaR < config.deltaRMin) {
            return false;
        }
        float cotTheta = (sp2.z() - sp1.z()) / deltaR;
        if (std::fabs(cotTheta) > config.cotThetaMax) {
            return false;
        }
        float zOrigin = sp1.z() - sp1.radius() * cotTheta;
        if (zOrigin < config.collisionRegionMin ||
            zOrigin > config.collisionRegionMax) {
            return false;
        }
    }
    return true;
}

lin_circle doublet_finding_helper::transform_coordinates(
    const internal_spacepoint<spacepoint>& sp1,
    const internal_spacepoint<spacepoint>& sp2, bool bottom) {
    const float& xM = sp1.x();
    const float& yM = sp1.y();
    const float& zM = sp1.z();
    const float& rM = sp1.radius();
    const float& varianceZM = sp1.varianceZ();
    const float& varianceRM = sp1.varianceR();
    float cosPhiM = xM / rM;
    float sinPhiM = yM / rM;

    float deltaX = sp2.x() - xM;
    float deltaY = sp2.y() - yM;
    float deltaZ = sp2.z() - zM;
    // calculate projection fraction of spM->sp vector pointing in same
    // direction as
    // vector origin->spM (x) and projection fraction of spM->sp vector pointing
    // orthogonal to origin->spM (y)
    float x = deltaX * cosPhiM + deltaY * sinPhiM;
    float y = deltaY * cosPhiM - deltaX * sinPhiM;
    // 1/(length of M -> SP)
    float iDeltaR2 = 1. / (deltaX * deltaX + deltaY * deltaY);
    float iDeltaR = std::sqrt(iDeltaR2);
    //
    int bottomFactor = 1 * (int(!bottom)) - 1 * (int(bottom));
    // cot_theta = (deltaZ/deltaR)
    float cot_theta = deltaZ * iDeltaR * bottomFactor;
    // VERY frequent (SP^3) access
    lin_circle l;
    l.m_cotTheta = cot_theta;
    // location on z-axis of this SP-duplet
    l.m_Zo = zM - rM * cot_theta;
    l.m_iDeltaR = iDeltaR;
    // transformation of circle equation (x,y) into linear equation (u,v)
    // x^2 + y^2 - 2x_0*x - 2y_0*y = 0
    // is transformed into
    // 1 - 2x_0*u - 2y_0*v = 0
    // using the following m_U and m_V
    // (u = A + B*v); A and B are created later on
    l.m_U = x * iDeltaR2;
    l.m_V = y * iDeltaR2;
    // error term for sp-pair without correlation of middle space point
    l.m_Er = ((varianceZM + sp2.varianceZ()) +
              (cot_theta * cot_theta) * (varianceRM + sp2.varianceR())) *
             iDeltaR2;

    return l;
}

}  // namespace traccc
