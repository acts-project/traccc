/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "utils/axis.hpp"
#include "utils/grid.hpp"

namespace traccc{

// define spacepoint_grid
using spacepoint_grid = grid<traccc::axis<AxisBoundaryType::Closed>,
			     traccc::axis<AxisBoundaryType::Bound> >;
    
struct seedfinder_config{

    // Seed Cuts
    // lower cutoff for seeds in MeV
    // FIXME: Acts units
    float minPt = 400.;
    // cot of maximum theta angle
    // equivalent to 2.7 eta (pseudorapidity)
    float cotThetaMax = 7.40627;
    // minimum distance in mm in r between two measurements within one seed
    float deltaRMin = 5;
    // maximum distance in mm in r between two measurements within one seed
    float deltaRMax = 270;
    
    // FIXME: this is not used yet
    //        float upperPtResolutionPerSeed = 20* Acts::GeV;
    
    // the delta for inverse helix radius up to which compared seeds
    // are considered to have a compatible radius. delta of inverse radius
    // leads to this value being the cutoff. unit is 1/mm. default value
    // of 0.00003 leads to all helices with radius>33m to be considered compatible
    
    // impact parameter in mm
    // FIXME: Acts units
    float impactMax = 20.;
    
    // how many sigmas of scattering angle should be considered?
    float sigmaScattering = 5;
    // Upper pt limit for scattering calculation
    float maxPtScattering = 10000;
    
    // for how many seeds can one SpacePoint be the middle SpacePoint?
    int maxSeedsPerSpM = 5;
    
    // Geometry Settings
    // Detector ROI
    // limiting location of collision region in z
    float collisionRegionMin = -150;
    float collisionRegionMax = +150;
    float phiMin = -M_PI;
    float phiMax = M_PI;
    // limiting location of measurements
    float zMin = -2800;
    float zMax = 2800;
    float rMax = 600;
    // WARNING: if rMin is smaller than impactMax, the bin size will be 2*pi,
    // which will make seeding very slow!
    float rMin = 33;
    
    // Unit in kiloTesla
    // FIXME: Acts units
    float bFieldInZ = 0.00208;
    // location of beam in x,y plane.
    // used as offset for Space Points
    vector2 beamPos{0, 0};
    
    // average radiation lengths of material on the length of a seed. used for
    // scattering.
    // default is 5%
    // TODO: necessary to make amount of material dependent on detector region?
    float radLengthPerSeed = 0.05;
    // alignment uncertainties, used for uncertainties in the
    // non-measurement-plane of the modules
    // which otherwise would be 0
    // will be added to spacepoint measurement uncertainties (and therefore also
    // multiplied by sigmaError)
    // FIXME: call align1 and align2
    float zAlign = 0;
    float rAlign = 0;
    // used for measurement (+alignment) uncertainties.
    // find seeds within 5sigma error ellipse
    float sigmaError = 5;
    
    // derived values, set on Seedfinder construction
    float highland = 0;
    float maxScatteringAngle2 = 0;
    float pTPerHelixRadius = 0;
    float minHelixDiameter2 = 0;
    float pT2perRadius = 0;    
};

// spacepoint grid configuration
struct spacepoint_grid_config {
    // magnetic field in kTesla
    float bFieldInZ;
    // minimum pT to be found by seedfinder in MeV
    float minPt;
    // maximum extension of sensitive detector layer relevant for seeding as
    // distance from x=y=0 (i.e. in r) in mm
    float rMax;
    // maximum extension of sensitive detector layer relevant for seeding in
    // positive direction in z in mm
    float zMax;
    // maximum extension of sensitive detector layer relevant for seeding in
    // negative direction in z in mm
    float zMin;
    // maximum distance in r from middle space point to bottom or top spacepoint
    // in mm
    float deltaRMax;
    // maximum forward direction expressed as cot(theta)
    float cotThetaMax;
};
    
} // namespace traccc
