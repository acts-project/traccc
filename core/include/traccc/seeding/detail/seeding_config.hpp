/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

namespace traccc {

struct seedfinder_config {

    seedfinder_config() { setup(); }

    // limiting location of measurements
    // Beomki's note: this value introduces redundant bins
    // without any spacepoints
    // m_config.zMin = -2800.;
    // m_config.zMax = 2800.;
    scalar zMin = -1186 * unit<scalar>::mm;
    scalar zMax = 1186 * unit<scalar>::mm;
    scalar rMax = 200 * unit<scalar>::mm;
    // WARNING: if rMin is smaller than impactMax, the bin size will be 2*pi,
    // which will make seeding very slow!
    scalar rMin = 33 * unit<scalar>::mm;

    // Geometry Settings
    // Detector ROI
    // limiting location of collision region in z
    scalar collisionRegionMin = -250 * unit<scalar>::mm;
    scalar collisionRegionMax = +250 * unit<scalar>::mm;
    scalar phiMin = static_cast<scalar>(-M_PI);
    scalar phiMax = static_cast<scalar>(M_PI);

    // Seed Cuts
    // lower cutoff for seeds in MeV
    scalar minPt = 500.f * unit<scalar>::MeV;
    // cot of maximum theta angle
    // equivalent to 2.7 eta (pseudorapidity)
    scalar cotThetaMax = 7.40627f;
    // minimum distance in mm in r between two measurements within one seed
    scalar deltaRMin = 1 * unit<scalar>::mm;
    // maximum distance in mm in r between two measurements within one seed
    scalar deltaRMax = 60 * unit<scalar>::mm;

    // FIXME: this is not used yet
    //        scalar upperPtResolutionPerSeed = 20* Acts::GeV;

    // the delta for inverse helix radius up to which compared seeds
    // are considered to have a compatible radius. delta of inverse radius
    // leads to this value being the cutoff. unit is 1/mm. default value
    // of 0.00003 leads to all helices with radius>33m to be considered
    // compatible

    // impact parameter in mm
    scalar impactMax = 10. * unit<scalar>::mm;
    // how many sigmas of scattering angle should be considered?
    scalar sigmaScattering = 1.0;
    // Upper pt limit for scattering calculation
    scalar maxPtScattering = 10 * unit<scalar>::GeV;

    // for how many seeds can one SpacePoint be the middle SpacePoint?
    int maxSeedsPerSpM = 20;

    scalar bFieldInZ = 1.99724f * unit<scalar>::T;
    // location of beam in x,y plane.
    // used as offset for Space Points
    vector2 beamPos{-.0 * unit<scalar>::mm, -.0 * unit<scalar>::mm};

    // average radiation lengths of material on the length of a seed. used for
    // scattering.
    // default is 5%
    // TODO: necessary to make amount of material dependent on detector region?
    scalar radLengthPerSeed = 0.05f;
    // alignment uncertainties, used for uncertainties in the
    // non-measurement-plane of the modules
    // which otherwise would be 0
    // will be added to spacepoint measurement uncertainties (and therefore also
    // multiplied by sigmaError)
    // FIXME: call align1 and align2
    scalar zAlign = 0 * unit<scalar>::mm;
    scalar rAlign = 0 * unit<scalar>::mm;
    // used for measurement (+alignment) uncertainties.
    // find seeds within 5sigma error ellipse
    scalar sigmaError = 5;

    // derived values, set on Seedfinder construction
    scalar highland = 0;
    scalar maxScatteringAngle2 = 0;
    scalar pTPerHelixRadius = 0;
    scalar minHelixDiameter2 = 0;
    scalar pT2perRadius = 0;

    // Multiplicator for the number of phi-bins. The minimum number of phi-bins
    // depends on min_pt, magnetic field: 2*M_PI/(minPT particle
    // phi-deflection). phiBinDeflectionCoverage is a multiplier for this
    // number. If numPhiNeighbors (in the configuration of the BinFinders) is
    // configured to return 1 neighbor on either side of the current phi-bin
    // (and you want to cover the full phi-range of minPT), leave this at 1.
    int phiBinDeflectionCoverage = 1;

    darray<unsigned int, 2> neighbor_scope{1, 1};

    TRACCC_HOST_DEVICE
    size_t get_num_rbins() const {
        return static_cast<size_t>(rMax + getter::norm(beamPos));
    }

    TRACCC_HOST_DEVICE
    unsigned int get_max_neighbor_bins() const {
        unsigned int t = neighbor_scope[0] + neighbor_scope[1] + 1;
        return t * t;
    }

    // Configure unset parameters
    TRACCC_HOST_DEVICE
    void setup() {
        highland = 13.6f * traccc::unit<traccc::scalar>::MeV *
                   std::sqrt(radLengthPerSeed) *
                   (1.f + 0.038f * std::log(radLengthPerSeed));

        float maxScatteringAngle = highland / minPt;
        maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;

        pTPerHelixRadius = bFieldInZ;
        minHelixDiameter2 = std::pow(minPt * 2.f / pTPerHelixRadius, 2.f);

        // @TODO: This is definitely a bug because highland / pTPerHelixRadius
        // is in length unit
        pT2perRadius = std::pow(highland / pTPerHelixRadius, 2.f);
    }
};

// spacepoint grid configuration
struct spacepoint_grid_config {

    spacepoint_grid_config() = delete;
    spacepoint_grid_config(const seedfinder_config& finder_config)
        : bFieldInZ(finder_config.bFieldInZ),
          minPt(finder_config.minPt),
          rMax(finder_config.rMax),
          zMax(finder_config.zMax),
          zMin(finder_config.zMin),
          deltaRMax(finder_config.deltaRMax),
          cotThetaMax(finder_config.cotThetaMax),
          impactMax(finder_config.impactMax),
          phiMin(finder_config.phiMin),
          phiMax(finder_config.phiMax),
          phiBinDeflectionCoverage(finder_config.phiBinDeflectionCoverage) {}

    // magnetic field in kTesla
    scalar bFieldInZ;
    // minimum pT to be found by seedfinder in MeV
    scalar minPt;
    // maximum extension of sensitive detector layer relevant for seeding as
    // distance from x=y=0 (i.e. in r) in mm
    scalar rMax;
    // maximum extension of sensitive detector layer relevant for seeding in
    // positive direction in z in mm
    scalar zMax;
    // maximum extension of sensitive detector layer relevant for seeding in
    // negative direction in z in mm
    scalar zMin;
    // maximum distance in r from middle space point to bottom or top spacepoint
    // in mm
    scalar deltaRMax;
    // maximum forward direction expressed as cot(theta)
    scalar cotThetaMax;
    // impact parameter in mm
    scalar impactMax;
    // minimum phi value for phiAxis construction
    scalar phiMin = static_cast<scalar>(-M_PI);
    // maximum phi value for phiAxis construction
    scalar phiMax = static_cast<scalar>(M_PI);
    // Multiplicator for the number of phi-bins. The minimum number of phi-bins
    // depends on min_pt, magnetic field: 2*M_PI/(minPT particle
    // phi-deflection). phiBinDeflectionCoverage is a multiplier for this
    // number. If numPhiNeighbors (in the configuration of the BinFinders) is
    // configured to return 1 neighbor on either side of the current phi-bin
    // (and you want to cover the full phi-range of minPT), leave this at 1.
    int phiBinDeflectionCoverage = 1;
};

struct seedfilter_config {
    // the allowed delta between two inverted seed radii for them to be
    // considered compatible.
    scalar deltaInvHelixDiameter = 0.00003f / unit<scalar>::mm;
    // the impact parameters (d0) is multiplied by this factor and subtracted
    // from weight
    scalar impactWeightFactor = 1.f;
    // seed weight increased by this value if a compatible seed has been found.
    scalar compatSeedWeight = 200.f;
    // minimum distance between compatible seeds to be considered for weight
    // boost
    scalar deltaRMin = 5.f * unit<scalar>::mm;
    // in dense environments many seeds may be found per middle space point.
    // only seeds with the highest weight will be kept if this limit is reached.
    unsigned int maxSeedsPerSpM = 20;
    // how often do you want to increase the weight of a seed for finding a
    // compatible seed?
    size_t compatSeedLimit = 2;
    // Tool to apply experiment specific cuts on collected middle space points

    size_t max_triplets_per_spM = 5;

    // seed weight increase
    scalar good_spB_min_radius = 150.f * unit<scalar>::mm;
    scalar good_spB_weight_increase = 400.f;
    scalar good_spT_max_radius = 150.f * unit<scalar>::mm;
    scalar good_spT_weight_increase = 200.f;

    // bottom sp cut
    scalar good_spB_min_weight = 380.f;

    // seed cut
    scalar seed_min_weight = 200.f;
    scalar spB_min_radius = 43.f * unit<scalar>::mm;
};

}  // namespace traccc
