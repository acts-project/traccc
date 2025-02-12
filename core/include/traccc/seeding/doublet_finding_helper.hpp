/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/doublet.hpp"
#include "traccc/seeding/detail/lin_circle.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_type.hpp"

namespace traccc {

// helper functions used for both cpu and gpu
struct doublet_finding_helper {
    /// Check if two spacepoints form doublets
    ///
    /// @param sp1 is middle spacepoint
    /// @param sp2 is bottom or top spacepoint
    /// @param config is configuration parameter
    /// @tparam otherSpType is whether it is for middle-bottom or middle-top
    /// doublet
    ///
    /// @return boolean value for compatibility
    ///
    template <details::spacepoint_type otherSpType, typename T1, typename T2>
    static inline TRACCC_HOST_DEVICE bool isCompatible(
        const edm::spacepoint<T1>& sp1, const edm::spacepoint<T2>& sp2,
        const seedfinder_config& config);

    /// Do the conformal transformation on doublet's coordinate
    ///
    /// @param sp1 is middle spacepoint
    /// @param sp2 is bottom or top spacepoint
    /// @tparam otherSpType is whether it is for middle-bottom or middle-top
    /// doublet
    ///
    /// @return lin_circle which contains the transformed coordinate information
    ///
    template <details::spacepoint_type otherSpType, typename T1, typename T2>
    static inline TRACCC_HOST_DEVICE lin_circle transform_coordinates(
        const edm::spacepoint<T1>& sp1, const edm::spacepoint<T2>& sp2);
};

template <details::spacepoint_type otherSpType, typename T1, typename T2>
bool TRACCC_HOST_DEVICE doublet_finding_helper::isCompatible(
    const edm::spacepoint<T1>& sp1, const edm::spacepoint<T2>& sp2,
    const seedfinder_config& config) {

    static_assert(otherSpType == details::spacepoint_type::bottom ||
                  otherSpType == details::spacepoint_type::top);

    scalar deltaR, cotTheta, zOrigin;
    if constexpr (otherSpType == details::spacepoint_type::bottom) {
        // check if R distance is too small, because bins are not R-sorted
        deltaR = sp1.radius() - sp2.radius();
        // actually cotTheta * deltaR to avoid division by 0 statements
        cotTheta = sp1.z() - sp2.z();
        // actually zOrigin * deltaR to avoid division by 0 statements
        zOrigin = sp1.z() * deltaR - sp1.radius() * cotTheta;
    } else {
        // check if R distance is too small, because bins are not R-sorted
        deltaR = sp2.radius() - sp1.radius();
        // actually cotTheta * deltaR to avoid division by 0 statements
        cotTheta = (sp2.z() - sp1.z());
        // actually zOrigin * deltaR to avoid division by 0 statements
        zOrigin = sp1.z() * deltaR - sp1.radius() * cotTheta;
    }
    return ((deltaR < config.deltaRMax) && (deltaR > config.deltaRMin) &&
            (math::fabs(cotTheta) < config.cotThetaMax * deltaR) &&
            (zOrigin > config.collisionRegionMin * deltaR) &&
            (zOrigin < config.collisionRegionMax * deltaR));
}

template <details::spacepoint_type otherSpType, typename T1, typename T2>
lin_circle TRACCC_HOST_DEVICE doublet_finding_helper::transform_coordinates(
    const edm::spacepoint<T1>& sp1, const edm::spacepoint<T2>& sp2) {

    static_assert(otherSpType == details::spacepoint_type::bottom ||
                  otherSpType == details::spacepoint_type::top);

    const scalar& xM = sp1.x();
    const scalar& yM = sp1.y();
    const scalar& zM = sp1.z();
    const scalar& rM = sp1.radius();
    const scalar& varianceZM = sp1.z_variance();
    const scalar& varianceRM = sp1.radius_variance();
    scalar cosPhiM = xM / rM;
    scalar sinPhiM = yM / rM;

    scalar deltaX = sp2.x() - xM;
    scalar deltaY = sp2.y() - yM;
    scalar deltaZ = sp2.z() - zM;
    // calculate projection fraction of spM->sp vector pointing in same
    // direction as
    // vector origin->spM (x) and projection fraction of spM->sp vector pointing
    // orthogonal to origin->spM (y)
    scalar x = deltaX * cosPhiM + deltaY * sinPhiM;
    scalar y = deltaY * cosPhiM - deltaX * sinPhiM;
    // 1/(length of M -> SP)
    scalar iDeltaR2 =
        static_cast<scalar>(1.) / (deltaX * deltaX + deltaY * deltaY);
    scalar iDeltaR = std::sqrt(iDeltaR2);
    // cot_theta = (deltaZ/deltaR)
    scalar cot_theta = deltaZ * iDeltaR;
    if constexpr (otherSpType == details::spacepoint_type::bottom) {
        cot_theta = -cot_theta;
    }
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
    l.m_Er = ((varianceZM + sp2.z_variance()) +
              (cot_theta * cot_theta) * (varianceRM + sp2.radius_variance())) *
             iDeltaR2;

    return l;
}

}  // namespace traccc
