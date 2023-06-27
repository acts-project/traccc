/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// io
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_spacepoints.hpp"

// algorithms
#include "traccc/seeding/seed_finding.hpp"
#include "traccc/seeding/spacepoint_binning.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// tests
#include "tests/atlas_cuts.hpp"
#include "tests/space_point.hpp"

// acts
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Seeding/BinFinder.hpp"
#include "Acts/Seeding/BinnedSPGroup.hpp"
#include "Acts/Seeding/EstimateTrackParamsFromSeed.hpp"
#include "Acts/Seeding/InternalSeed.hpp"
#include "Acts/Seeding/InternalSpacePoint.hpp"
#include "Acts/Seeding/Seed.hpp"
#include "Acts/Seeding/SeedFilter.hpp"
#include "Acts/Seeding/SeedFinder.hpp"
#include "Acts/Seeding/SpacePointGrid.hpp"
#include "Acts/Surfaces/DiscSurface.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"

// VecMem
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

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
                 traccc::getter::element(traccc_vec, traccc::e_bound_loc0, 0)) <
            traccc::float_epsilon * 10 &&
        std::abs(acts_vec[Acts::eBoundLoc1] -
                 traccc::getter::element(traccc_vec, traccc::e_bound_loc1, 0)) <
            traccc::float_epsilon * 10 &&
        std::abs(acts_vec[Acts::eBoundTheta] -
                 traccc::getter::element(traccc_vec, traccc::e_bound_theta,
                                         0)) < traccc::float_epsilon * 1000 &&
        std::abs(acts_vec[Acts::eBoundPhi] -
                 traccc::getter::element(traccc_vec, traccc::e_bound_phi, 0)) <
            traccc::float_epsilon * 1000) {
        return true;
    }
    return false;
}

class CompareWithActsSeedingTests
    : public ::testing::TestWithParam<
          std::tuple<std::string, std::string, unsigned int>> {};

// This defines the local frame test suite
TEST_P(CompareWithActsSeedingTests, Run) {

    std::string detector_file = std::get<0>(GetParam());
    std::string hits_dir = std::get<1>(GetParam());
    unsigned int event = std::get<2>(GetParam());

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Seeding Config
    traccc::seedfinder_config traccc_config;
    traccc::spacepoint_grid_config grid_config(traccc_config);

    // Declare algorithms
    traccc::spacepoint_binning sb(traccc_config, grid_config, host_mr);
    traccc::seed_finding sf(traccc_config, traccc::seedfilter_config());
    traccc::track_params_estimation tp(host_mr);

    // Read the surface transforms
    auto surface_transforms = traccc::io::read_geometry(detector_file);

    // Read the hits from the relevant event file
    traccc::io::spacepoint_reader_output reader_output(&host_mr);
    traccc::io::read_spacepoints(reader_output, event, hits_dir,
                                 surface_transforms, traccc::data_format::csv);

    traccc::spacepoint_collection_types::host& spacepoints_per_event =
        reader_output.spacepoints;
    traccc::cell_module_collection_types::host& modules_per_event =
        reader_output.modules;
    /*--------------------------------
      TRACCC seeding
      --------------------------------*/

    auto internal_spacepoints_per_event = sb(spacepoints_per_event);
    auto seeds = sf(spacepoints_per_event, internal_spacepoints_per_event);

    /*--------------------------------
      TRACCC track params estimation
      --------------------------------*/

    auto tp_output = tp(spacepoints_per_event, seeds, modules_per_event,
                        {0.f, 0.f, traccc_config.bFieldInZ});
    auto& traccc_params = tp_output;

    /*--------------------------------
      ACTS seeding
      --------------------------------*/

    // copy traccc::spacepoint into SpacePoint
    std::vector<const SpacePoint*> spVec;
    for (auto& sp : spacepoints_per_event) {
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

    // spacepoint equality check
    int n_sp_match = 0;
    for (auto& sp : spacepoints_per_event) {
        if (std::find(spVec.begin(), spVec.end(), sp) != spVec.end()) {
            n_sp_match++;
        }
    }
    EXPECT_EQ(spacepoints_per_event.size(), n_sp_match);
    EXPECT_EQ(spVec.size(), n_sp_match);

    Acts::SeedFinderConfig<SpacePoint> acts_config;

    // silicon detector max
    acts_config.phiMin = traccc_config.phiMin;
    acts_config.phiMax = traccc_config.phiMax;

    acts_config.rMin = traccc_config.rMin;
    acts_config.rMax = traccc_config.rMax;
    acts_config.deltaRMin = traccc_config.deltaRMin;
    acts_config.deltaRMinTopSP = traccc_config.deltaRMin;
    acts_config.deltaRMinBottomSP = traccc_config.deltaRMin;
    acts_config.deltaRMax = traccc_config.deltaRMax;
    acts_config.deltaRMaxTopSP = traccc_config.deltaRMax;
    acts_config.deltaRMaxBottomSP = traccc_config.deltaRMax;
    acts_config.collisionRegionMin = traccc_config.collisionRegionMin;
    acts_config.collisionRegionMax = traccc_config.collisionRegionMax;

    acts_config.zMin = traccc_config.zMin;
    acts_config.zMax = traccc_config.zMax;
    acts_config.maxSeedsPerSpM = traccc_config.maxSeedsPerSpM;

    // 2.7 eta
    acts_config.cotThetaMax = traccc_config.cotThetaMax;
    acts_config.sigmaScattering = traccc_config.sigmaScattering;
    acts_config.maxPtScattering = traccc_config.maxPtScattering;

    acts_config.minPt = traccc_config.minPt;
    acts_config.bFieldInZ = traccc_config.bFieldInZ;

    acts_config.beamPos[0] = traccc_config.beamPos[0];
    acts_config.beamPos[1] = traccc_config.beamPos[1];

    acts_config.impactMax = traccc_config.impactMax;

    acts_config.sigmaError = traccc_config.sigmaError;

    auto bottomBinFinder = std::make_shared<Acts::BinFinder<SpacePoint>>(
        Acts::BinFinder<SpacePoint>());
    auto topBinFinder = std::make_shared<Acts::BinFinder<SpacePoint>>(
        Acts::BinFinder<SpacePoint>());
    Acts::SeedFilterConfig sfconf;
    sfconf.maxSeedsPerSpM = traccc::seedfilter_config().maxSeedsPerSpM;

    Acts::ATLASCuts<SpacePoint> atlasCuts = Acts::ATLASCuts<SpacePoint>();
    acts_config.seedFilter = std::make_unique<Acts::SeedFilter<SpacePoint>>(
        Acts::SeedFilter<SpacePoint>(sfconf, &atlasCuts));
    Acts::SeedFinder<SpacePoint> a(acts_config);

    // covariance tool, sets covariances per spacepoint as required
    auto ct = [=](const SpacePoint& sp, float, float,
                  float) -> std::pair<Acts::Vector3, Acts::Vector2> {
        Acts::Vector3 position(sp.x(), sp.y(), sp.z());
        Acts::Vector2 covariance(sp.varianceR, sp.varianceZ);
        return std::make_pair(position, covariance);
    };

    // setup spacepoint grid config
    Acts::SpacePointGridConfig gridConf;

    gridConf.bFieldInZ = acts_config.bFieldInZ;
    gridConf.minPt = acts_config.minPt;
    gridConf.rMax = acts_config.rMax;
    gridConf.zMax = acts_config.zMax;
    gridConf.zMin = acts_config.zMin;
    gridConf.deltaRMax = acts_config.deltaRMax;
    gridConf.cotThetaMax = acts_config.cotThetaMax;
    gridConf.impactMax = acts_config.impactMax;
    gridConf.phiMin = acts_config.phiMin;
    gridConf.phiMax = acts_config.phiMax;
    gridConf.phiBinDeflectionCoverage = traccc_config.phiBinDeflectionCoverage;

    // create grid with bin sizes according to the configured geometry
    std::unique_ptr<Acts::SpacePointGrid<SpacePoint>> grid =
        Acts::SpacePointGridCreator::createGrid<SpacePoint>(gridConf);

    // Currently traccc is using grid2. check if acts grid dimensionality is the
    // same
    EXPECT_EQ(grid->numLocalBins().size(), 2);

    detray::axis::circular axis0 = internal_spacepoints_per_event.axis_p0();
    detray::axis::regular axis1 = internal_spacepoints_per_event.axis_p1();

    auto acts_axis0 = grid->axes()[0];
    auto acts_axis1 = grid->axes()[1];

    EXPECT_EQ(axis0.bins(), acts_axis0->getNBins());
    EXPECT_EQ(axis1.bins(), acts_axis1->getNBins());

    auto axis0_borders = axis0.all_borders();
    auto axis1_borders = axis1.all_borders();
    auto acts_axis0_borders = acts_axis0->getBinEdges();
    auto acts_axis1_borders = acts_axis1->getBinEdges();

    EXPECT_EQ(axis0_borders.size(), axis0.bins() + 1);
    EXPECT_EQ(axis1_borders.size(), axis1.bins() + 1);

    EXPECT_EQ(axis0_borders.size(), acts_axis0_borders.size());
    EXPECT_EQ(axis1_borders.size(), acts_axis1_borders.size());

    for (unsigned int i = 0; i < axis0.bins() + 1; ++i) {
        EXPECT_NEAR(axis0_borders[i], acts_axis0_borders[i], 0.01);
    }
    for (unsigned int i = 0; i < axis1.bins() + 1; ++i) {
        EXPECT_NEAR(axis1_borders[i], acts_axis1_borders[i], 0.01);
    }

    auto spGroup = Acts::BinnedSPGroup<SpacePoint>(
        spVec.begin(), spVec.end(), ct, bottomBinFinder, topBinFinder,
        std::move(grid), Acts::Extent(), acts_config);

    auto groupIt = spGroup.begin();
    auto endOfGroups = spGroup.end();

    // Run the ACTS seeding
    std::vector<Acts::Seed<SpacePoint>> seedVector;
    for (; !(groupIt == endOfGroups); ++groupIt) {
        auto seed_group = a.createSeedsForGroup(
            groupIt.bottom(), groupIt.middle(), groupIt.top());
        seedVector.insert(seedVector.end(), seed_group.begin(),
                          seed_group.end());
    }

    // Count the number of matching seeds
    // and push_back seed into sorted_seedVector
    std::vector<Acts::Seed<SpacePoint>> sorted_seedVector;
    int n_seed_match = 0;
    for (auto& seed : seeds) {
        auto it = std::find_if(
            seedVector.begin(), seedVector.end(), [&](auto acts_seed) {
                auto traccc_spB = spacepoints_per_event.at(seed.spB_link);
                auto traccc_spM = spacepoints_per_event.at(seed.spM_link);
                auto traccc_spT = spacepoints_per_event.at(seed.spT_link);

                auto& triplets = acts_seed.sp();
                auto& acts_spB = triplets[0];
                auto& acts_spM = triplets[1];
                auto& acts_spT = triplets[2];

                if (acts_spB == traccc_spB && acts_spM == traccc_spM &&
                    acts_spT == traccc_spT) {
                    return true;
                }

                return false;
            });

        if (it != seedVector.end()) {
            sorted_seedVector.push_back(*it);
            n_seed_match++;
        }
    }
    seedVector = sorted_seedVector;

    float seed_match_ratio = float(n_seed_match) / seeds.size();

    // Ensure that ACTS and traccc give the same result
    // @TODO Uncomment the line below once acts-project/acts#2132 is merged
    // EXPECT_EQ(seeds.size(), seedVector.size());
    EXPECT_NEAR(seeds.size(), seedVector.size(), seeds.size() * 0.001);
    EXPECT_TRUE(seed_match_ratio > 0.999);

    /*--------------------------------
      ACTS track params estimation
      --------------------------------*/

    const Acts::GeometryContext geoCtx;

    std::vector<Acts::BoundVector> acts_params;

    for (auto& seed : seedVector) {

        auto& spacePoints = seed.sp();

        // get SpacePointPtr
        std::array<const SpacePoint*, 3> spacePointPtrs;
        spacePointPtrs[0] = spacePoints[0];
        spacePointPtrs[1] = spacePoints[1];
        spacePointPtrs[2] = spacePoints[2];

        // find geometry id
        auto spB = spacePoints[0];
        traccc::geometry_id geo_id = 0;
        auto it = std::find(spacepoints_per_event.begin(),
                            spacepoints_per_event.end(), spB);
        if (it != spacepoints_per_event.end()) {
            geo_id = modules_per_event.at((*it).meas.module_link).module;
        }

        EXPECT_TRUE(geo_id != 0);

        const auto& tf3 = surface_transforms[geo_id];
        const auto& tsl = tf3.translation();
        const auto& rot = tf3.rotation();

        Acts::Vector3 normal;
        normal(0, 0) = traccc::transform3::element_getter()(rot, 0, 2);
        normal(1, 0) = traccc::transform3::element_getter()(rot, 1, 2);
        normal(2, 0) = traccc::transform3::element_getter()(rot, 2, 2);

        std::shared_ptr<Acts::Surface> bottomSurface;
        bool is_disc = false;
        // barrel layer
        if (abs(normal.dot(Acts::Vector3::UnitZ())) < traccc::float_epsilon) {
            // for plane of barrel layers, translation and normal vector is used
            // to form acts transform3

            Acts::Vector3 center;
            center(0, 0) = tsl[0];
            center(1, 0) = tsl[1];
            center(2, 0) = tsl[2];

            bottomSurface =
                Acts::Surface::makeShared<Acts::PlaneSurface>(center, normal);
        }
        // endcap layer
        else {
            is_disc = true;
            // for disc of endcap layers, the traccc transform components are
            // copied into acts transform3
            Acts::Transform3 acts_tf3;
            for (unsigned int i = 0; i < 4; i++) {
                for (unsigned int j = 0; j < 4; j++) {
                    acts_tf3(i, j) = traccc::transform3::element_getter()(
                        tf3.matrix(), i, j);
                }
            }

            // last three arugments are given randomly
            bottomSurface = Acts::Surface::makeShared<Acts::DiscSurface>(
                acts_tf3, 0., 10., 0.);
        }

        // Test the full track parameters estimator
        auto fullParamsOpt = estimateTrackParamsFromSeed(
            geoCtx, spacePointPtrs.begin(), spacePointPtrs.end(),
            *bottomSurface,
            Acts::Vector3(
                0, 0, acts_config.bFieldInZ / traccc::unit<traccc::scalar>::T),
            0.1);

        auto acts_vec = *fullParamsOpt;

        // Acts globalToLocal function on DiscSurface gives (u,v) in radial
        // coordinate. Therefore acts parameters are converted into cartesian
        // coordinate for comparison with traccc parameters
        if (is_disc) {
            auto x = acts_vec[Acts::eBoundLoc0] *
                     std::cos(acts_vec[Acts::eBoundLoc1]);
            auto y = acts_vec[Acts::eBoundLoc0] *
                     std::sin(acts_vec[Acts::eBoundLoc1]);
            acts_vec[Acts::eBoundLoc0] = x;
            acts_vec[Acts::eBoundLoc1] = y;
        }

        acts_params.push_back(acts_vec);
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
    // @TODO Uncomment the line below once acts-project/acts#2132 is merged
    // EXPECT_EQ(acts_params.size(), traccc_params.size())
    EXPECT_NEAR(acts_params.size(), traccc_params.size(),
                acts_params.size() * 0.001);
    EXPECT_TRUE(params_match_ratio > 0.999)
        << "Parameter matching ratio: " << params_match_ratio << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    SeedingValidation, CompareWithActsSeedingTests,
    ::testing::Values(std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 0),
                      std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 1),
                      std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 2),
                      std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 3),
                      std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 4),
                      std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 5),
                      std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 6),
                      std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 7),
                      std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 8),
                      std::make_tuple("tml_detector/trackml-detector.csv",
                                      "tml_full/ttbar_mu200/", 9)));
