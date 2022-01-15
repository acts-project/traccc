/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <definitions/primitives.hpp>
#include <definitions/qualifiers.hpp>

namespace traccc {

struct seedfinder_config {
    // limiting location of measurements
    // Beomki's note: this value introduces redundant bins
    // without any spacepoints
    // m_config.zMin = -2800.;
    // m_config.zMax = 2800.;
    scalar zMin = -1186;
    scalar zMax = 1186;
    scalar rMax = 160;
    // WARNING: if rMin is smaller than impactMax, the bin size will be 2*pi,
    // which will make seeding very slow!
    scalar rMin = 33;

    // Geometry Settings
    // Detector ROI
    // limiting location of collision region in z
    scalar collisionRegionMin = -250;
    scalar collisionRegionMax = +250;
    scalar phiMin = -M_PI;
    scalar phiMax = M_PI;

    // Seed Cuts
    // lower cutoff for seeds in MeV
    // FIXME: Acts units
    scalar minPt = 500.;
    // cot of maximum theta angle
    // equivalent to 2.7 eta (pseudorapidity)
    scalar cotThetaMax = 7.40627;
    // minimum distance in mm in r between two measurements within one seed
    scalar deltaRMin = 5;
    // maximum distance in mm in r between two measurements within one seed
    scalar deltaRMax = 160;

    // FIXME: this is not used yet
    //        scalar upperPtResolutionPerSeed = 20* Acts::GeV;

    // the delta for inverse helix radius up to which compared seeds
    // are considered to have a compatible radius. delta of inverse radius
    // leads to this value being the cutoff. unit is 1/mm. default value
    // of 0.00003 leads to all helices with radius>33m to be considered
    // compatible

    // impact parameter in mm
    // FIXME: Acts units
    scalar impactMax = 10.;
    // how many sigmas of scattering angle should be considered?
    scalar sigmaScattering = 1.0;
    // Upper pt limit for scattering calculation
    scalar maxPtScattering = 10000;

    // for how many seeds can one SpacePoint be the middle SpacePoint?
    int maxSeedsPerSpM = 5;

    // Unit in kiloTesla
    // FIXME: Acts units
    scalar bFieldInZ = 0.00199724;
    // location of beam in x,y plane.
    // used as offset for Space Points
    vector2 beamPos{-.5, -.5};
    scalar beamPos_x = -0.5, beamPos_y = -0.5;

    // average radiation lengths of material on the length of a seed. used for
    // scattering.
    // default is 5%
    // TODO: necessary to make amount of material dependent on detector region?
    scalar radLengthPerSeed = 0.05;
    // alignment uncertainties, used for uncertainties in the
    // non-measurement-plane of the modules
    // which otherwise would be 0
    // will be added to spacepoint measurement uncertainties (and therefore also
    // multiplied by sigmaError)
    // FIXME: call align1 and align2
    scalar zAlign = 0;
    scalar rAlign = 0;
    // used for measurement (+alignment) uncertainties.
    // find seeds within 5sigma error ellipse
    scalar sigmaError = 5;

    // derived values, set on Seedfinder construction
    scalar highland = 0;
    scalar maxScatteringAngle2 = 0;
    scalar pTPerHelixRadius = 0;
    scalar minHelixDiameter2 = 0;
    scalar pT2perRadius = 0;

    darray<unsigned long, 2> neighbor_scope{1, 1};

    TRACCC_HOST_DEVICE
    size_t get_num_rbins() const {
        return static_cast<size_t>(rMax + getter::norm(beamPos));
    }

    TRACCC_HOST_DEVICE
    unsigned int get_max_neighbor_bins() const {
        return std::pow(neighbor_scope[0] + neighbor_scope[1] + 1, 2);
    }
};

// spacepoint grid configuration
struct spacepoint_grid_config {

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
};

struct seedfilter_config {
    // the allowed delta between two inverted seed radii for them to be
    // considered compatible.
    scalar deltaInvHelixDiameter = 0.00003;
    // the impact parameters (d0) is multiplied by this factor and subtracted
    // from weight
    scalar impactWeightFactor = 1.;
    // seed weight increased by this value if a compatible seed has been found.
    scalar compatSeedWeight = 200.;
    // minimum distance between compatible seeds to be considered for weight
    // boost
    scalar deltaRMin = 5.;
    // in dense environments many seeds may be found per middle space point.
    // only seeds with the highest weight will be kept if this limit is reached.
    unsigned int maxSeedsPerSpM = 10;
    // how often do you want to increase the weight of a seed for finding a
    // compatible seed?
    size_t compatSeedLimit = 2;
    // Tool to apply experiment specific cuts on collected middle space points

    size_t max_triplets_per_spM = 5;

    // seed weight increase
    scalar good_spB_min_radius = 150.;
    scalar good_spB_weight_increase = 400.;
    scalar good_spT_max_radius = 150.;
    scalar good_spT_weight_increase = 200.;

    // bottom sp cut
    scalar good_spB_min_weight = 380;

    // seed cut
    scalar seed_min_weight = 200;
    scalar spB_min_radius = 43.;
};

}  // namespace traccc
