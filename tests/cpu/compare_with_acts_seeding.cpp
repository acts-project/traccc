/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <iostream>

// io
#include "io/csv.hpp"
#include "io/reader.hpp"
#include "io/utils.hpp"
#include "io/writer.hpp"

// algorithms
#include "clusterization/clusterization_algorithm.hpp"
#include "seeding/track_params_estimation.hpp"
#include "track_finding/seeding_algorithm.hpp"

// Acts
#include "../../Tests/UnitTests/Core/Seeding/ATLASCuts.hpp"
#include "../../Tests/UnitTests/Core/Seeding/SpacePoint.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Seeding/BinFinder.hpp"
#include "Acts/Seeding/BinnedSPGroup.hpp"
#include "Acts/Seeding/EstimateTrackParamsFromSeed.hpp"
#include "Acts/Seeding/InternalSeed.hpp"
#include "Acts/Seeding/InternalSpacePoint.hpp"
#include "Acts/Seeding/Seed.hpp"
#include "Acts/Seeding/SeedFilter.hpp"
#include "Acts/Seeding/Seedfinder.hpp"
#include "Acts/Seeding/SpacePointGrid.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"

inline bool operator==(const SpacePoint* acts_sp,
                       const traccc::spacepoint& traccc_sp) {
    if (abs(acts_sp->x() - traccc_sp.global[0]) < traccc::float_epsilon &&
        abs(acts_sp->y() - traccc_sp.global[1]) < traccc::float_epsilon &&
        abs(acts_sp->z() - traccc_sp.global[2]) < traccc::float_epsilon) {
        return true;
    }
    return false;
}

inline bool operator==(const traccc::spacepoint& traccc_sp,
                       const SpacePoint* acts_sp) {
    if (abs(acts_sp->x() - traccc_sp.global[0]) < traccc::float_epsilon &&
        abs(acts_sp->y() - traccc_sp.global[1]) < traccc::float_epsilon &&
        abs(acts_sp->z() - traccc_sp.global[2]) < traccc::float_epsilon) {
        return true;
    }
    return false;
}

inline bool operator==(const Acts::BoundVector& acts_vec,
                       const traccc::bound_vector& traccc_vec) {
    if (std::abs(acts_vec[Acts::eBoundLoc0] -
                 traccc_vec[traccc::e_bound_loc0]) <
            traccc::float_epsilon * 10 &&
        std::abs(acts_vec[Acts::eBoundLoc1] -
                 traccc_vec[traccc::e_bound_loc1]) <
            traccc::float_epsilon * 10 &&
        std::abs(acts_vec[Acts::eBoundTheta] -
                 traccc_vec[traccc::e_bound_theta]) <
            traccc::float_epsilon * 10 &&
        std::abs(acts_vec[Acts::eBoundPhi] - traccc_vec[traccc::e_bound_phi]) <
            traccc::float_epsilon * 10) {
        return true;
    }
    return false;
}

inline bool operator==(const traccc::seed& rhs,
                       const Acts::Seed<SpacePoint>& lhs) {
    auto& triplets = lhs.sp();
    auto& acts_spB = triplets[0];
    auto& acts_spM = triplets[1];
    auto& acts_spT = triplets[2];

    auto& traccc_spB = rhs.spB;
    auto& traccc_spM = rhs.spM;
    auto& traccc_spT = rhs.spT;

    if (acts_spB == traccc_spB && acts_spM == traccc_spM &&
        acts_spT == traccc_spT) {
        return true;
    }

    return false;
}

// This defines the local frame test suite
TEST(algorithms, compare_with_acts_seeding) {

    std::string detector_file = "tml_detector/trackml-detector.csv";
    std::string hits_dir = "tml_hits/";
    unsigned int event = 0;

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    traccc::seeding_algorithm sa(host_mr);
    traccc::track_params_estimation tp(host_mr);

    // Read the surface transforms
    auto surface_transforms = traccc::read_geometry(detector_file);

    // Read the hits from the relevant event file
    traccc::host_spacepoint_container spacepoints_per_event =
        traccc::read_spacepoints_from_event(event, hits_dir, surface_transforms,
                                            host_mr);

    /*--------------------------------
      TRACCC seeding
      --------------------------------*/

    auto seeds = sa(spacepoints_per_event);

    /*--------------------------------
      TRACCC track params estimation
      --------------------------------*/

    auto tp_output = tp(seeds);
    auto& traccc_params = tp_output;

    /*--------------------------------
      ACTS seeding
      --------------------------------*/

    // copy traccc::spacepoint into SpacePoint
    std::vector<const SpacePoint*> spVec;
    for (std::size_t i_h = 0; i_h < spacepoints_per_event.size(); i_h++) {
        auto& items = spacepoints_per_event.get_items()[i_h];
        for (auto& sp : items) {

            SpacePoint* acts_sp =
                new SpacePoint{static_cast<float>(sp.global[0]),
                               static_cast<float>(sp.global[1]),
                               static_cast<float>(sp.global[2]),
                               std::hypot(static_cast<float>(sp.global[0]),
                                          static_cast<float>(sp.global[1])),
                               0,
                               0,
                               0};
            spVec.push_back(acts_sp);
        }
    }

    // spacepoint equality check
    int n_sp_match = 0;
    for (std::size_t i_h = 0; i_h < spacepoints_per_event.size(); i_h++) {
        auto& items = spacepoints_per_event.get_items()[i_h];
        for (auto& sp : items) {
            if (std::find(spVec.begin(), spVec.end(), sp) != spVec.end()) {
                n_sp_match++;
            }
        }
    }
    EXPECT_EQ(spacepoints_per_event.total_size(), n_sp_match);
    EXPECT_EQ(spVec.size(), n_sp_match);

    // Setup the config
    auto traccc_config = sa.get_seedfinder_config();
    // auto traccc_grid_config = sa.get_spacepoint_grid_config();

    Acts::SeedfinderConfig<SpacePoint> config;

    // silicon detector max
    config.phiMin = traccc_config.phiMin;
    config.phiMax = traccc_config.phiMax;

    config.rMin = traccc_config.rMin;
    config.rMax = traccc_config.rMax;
    config.deltaRMin = traccc_config.deltaRMin;
    config.deltaRMax = traccc_config.deltaRMax;
    config.collisionRegionMin = traccc_config.collisionRegionMin;
    config.collisionRegionMax = traccc_config.collisionRegionMax;

    config.zMin = traccc_config.zMin;
    config.zMax = traccc_config.zMax;
    config.maxSeedsPerSpM = traccc_config.maxSeedsPerSpM;

    // 2.7 eta
    config.cotThetaMax = traccc_config.cotThetaMax;
    config.sigmaScattering = traccc_config.sigmaScattering;
    config.maxPtScattering = traccc_config.maxPtScattering;

    config.minPt = traccc_config.minPt;
    config.bFieldInZ = traccc_config.bFieldInZ;

    config.beamPos[0] = traccc_config.beamPos[0];
    config.beamPos[1] = traccc_config.beamPos[1];

    config.impactMax = traccc_config.impactMax;

    config.sigmaError = traccc_config.sigmaError;

    auto bottomBinFinder = std::make_shared<Acts::BinFinder<SpacePoint>>(
        Acts::BinFinder<SpacePoint>());
    auto topBinFinder = std::make_shared<Acts::BinFinder<SpacePoint>>(
        Acts::BinFinder<SpacePoint>());
    Acts::SeedFilterConfig sfconf;
    Acts::ATLASCuts<SpacePoint> atlasCuts = Acts::ATLASCuts<SpacePoint>();
    config.seedFilter = std::make_unique<Acts::SeedFilter<SpacePoint>>(
        Acts::SeedFilter<SpacePoint>(sfconf, &atlasCuts));
    Acts::Seedfinder<SpacePoint> a(config);

    // covariance tool, sets covariances per spacepoint as required
    auto ct = [=](const SpacePoint& sp, float, float,
                  float) -> std::pair<Acts::Vector3, Acts::Vector2> {
        Acts::Vector3 position(sp.x(), sp.y(), sp.z());
        Acts::Vector2 covariance(sp.varianceR, sp.varianceZ);
        return std::make_pair(position, covariance);
    };

    // setup spacepoint grid config
    Acts::SpacePointGridConfig gridConf;
    gridConf.bFieldInZ = config.bFieldInZ;
    gridConf.minPt = config.minPt;
    gridConf.rMax = config.rMax;
    gridConf.zMax = config.zMax;
    gridConf.zMin = config.zMin;
    gridConf.deltaRMax = config.deltaRMax;
    gridConf.cotThetaMax = config.cotThetaMax;

    // create grid with bin sizes according to the configured geometry
    std::unique_ptr<Acts::SpacePointGrid<SpacePoint>> grid =
        Acts::SpacePointGridCreator::createGrid<SpacePoint>(gridConf);
    auto spGroup = Acts::BinnedSPGroup<SpacePoint>(
        spVec.begin(), spVec.end(), ct, bottomBinFinder, topBinFinder,
        std::move(grid), config);

    std::vector<std::vector<Acts::Seed<SpacePoint>>> seedVector;
    auto groupIt = spGroup.begin();
    auto endOfGroups = spGroup.end();

    for (; !(groupIt == endOfGroups); ++groupIt) {
        seedVector.push_back(a.createSeedsForGroup(
            groupIt.bottom(), groupIt.middle(), groupIt.top()));
    }

    // seed equality check
    int n_seed_match = 0;
    for (auto& outputVec : seedVector) {
        for (auto& seed : outputVec) {
            if (std::find(seeds.get_items()[0].begin(),
                          seeds.get_items()[0].end(),
                          seed) != seeds.get_items()[0].end()) {
                n_seed_match++;
            }
        }
    }

    float seed_match_ratio = float(n_seed_match) / seeds.total_size();
    EXPECT_TRUE((seed_match_ratio > 0.95) && (seed_match_ratio <= 1.));

    /*--------------------------------
      ACTS track params estimation
      --------------------------------*/
    const Acts::GeometryContext geoCtx;

    std::vector<Acts::BoundVector> acts_params;

    for (auto& outputVec : seedVector) {
        for (auto& seed : outputVec) {

            auto& spacePoints = seed.sp();

            // get SpacePointPtr
            std::array<const SpacePoint*, 3> spacePointPtrs;
            spacePointPtrs[0] = spacePoints[0];
            spacePointPtrs[1] = spacePoints[1];
            spacePointPtrs[2] = spacePoints[2];

            // find geometry id
            auto spB = spacePoints[0];
            traccc::geometry_id geo_id = 0;
            for (std::size_t i_h = 0; i_h < spacepoints_per_event.size();
                 i_h++) {
                auto& items = spacepoints_per_event.get_items()[i_h];
                if (std::find(items.begin(), items.end(), spB) != items.end()) {
                    geo_id = spacepoints_per_event.get_headers()[i_h];
                    break;
                }
            }

            EXPECT_TRUE(geo_id != 0);

            const auto& tf3 = surface_transforms[geo_id];
            const auto& tsl = tf3.translation();
            const auto& rot = tf3.rotation();

            Acts::Vector3 center;
            center(0, 0) = tsl[0];
            center(1, 0) = tsl[1];
            center(2, 0) = tsl[2];

            Acts::Vector3 normal;
            normal(0, 0) = traccc::transform3::element_getter()(rot, 0, 2);
            normal(1, 0) = traccc::transform3::element_getter()(rot, 1, 2);
            normal(2, 0) = traccc::transform3::element_getter()(rot, 2, 2);

            auto bottomSurface =
                Acts::Surface::makeShared<Acts::PlaneSurface>(center, normal);

            // Test the full track parameters estimator
            auto fullParamsOpt = estimateTrackParamsFromSeed(
                geoCtx, spacePointPtrs.begin(), spacePointPtrs.end(),
                *bottomSurface, Acts::Vector3(0, 0, 2), 0.1);

            acts_params.push_back(*fullParamsOpt);
        }
    }

    // params equality check
    int n_params_match = 0;
    for (auto& traccc_param : traccc_params) {
        auto& traccc_vec = traccc_param.vector();
        for (auto& acts_vec : acts_params) {
            if (acts_vec == traccc_vec) {
                n_params_match++;
                break;
            }
        }
    }

    float params_match_ratio = float(n_params_match) / traccc_params.size();
    EXPECT_TRUE((params_match_ratio > 0.95) && (params_match_ratio <= 1.));

    std::cout << "-------- Result ---------" << std::endl;
    std::cout << "seed matching ratio: " << seed_match_ratio << std::endl;
    std::cout << "params matching ratio: " << params_match_ratio << std::endl;
}
