/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// io
#include "traccc/io/read_spacepoints.hpp"

// algorithms
#include "traccc/seeding/detail/seed_finding.hpp"
#include "traccc/seeding/detail/spacepoint_binning.hpp"

// tests
#include "tests/atlas_cuts.hpp"

// acts
#include <Acts/EventData/Seed.hpp>
#include <Acts/EventData/SpacePointContainer.hpp>
#include <Acts/Geometry/GeometryContext.hpp>
#include <Acts/MagneticField/ConstantBField.hpp>
#include <Acts/Seeding/BinnedGroup.hpp>
#include <Acts/Seeding/EstimateTrackParamsFromSeed.hpp>
#include <Acts/Seeding/SeedFilter.hpp>
#include <Acts/Seeding/SeedFinder.hpp>
#include <Acts/Seeding/SpacePointGrid.hpp>
#include <Acts/Surfaces/DiscSurface.hpp>
#include <Acts/Surfaces/PlaneSurface.hpp>
#include <Acts/Surfaces/Surface.hpp>
#include <Acts/Utilities/GridBinFinder.hpp>
#include <Acts/Utilities/Helpers.hpp>

// VecMem
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <functional>
#include <limits>

// We need to define a 'SpacePointCollector' that will bridge
// between the ACTS internal EDM and the experiment EDM
// This class will accept a user-defined collection of space points and
// need to implement some _impl function to instruct ACTS how to retrieve the
// required quantities

// The internal details of this class are largely up to the user
class SpacePointCollector {
    public:
    using ActsSpacePointContainer =
        Acts::SpacePointContainer<SpacePointCollector, Acts::detail::RefHolder>;
    friend ActsSpacePointContainer;
    using ValueType =
        traccc::edm::spacepoint_collection::host::const_proxy_type;

    explicit SpacePointCollector(
        traccc::edm::spacepoint_collection::host& spacepoints)
        : m_storage(spacepoints) {}

    std::size_t size_impl() const { return m_storage.get().size(); }
    double x_impl(std::size_t idx) const { return m_storage.get()[idx].x(); }
    double y_impl(std::size_t idx) const { return m_storage.get()[idx].y(); }
    double z_impl(std::size_t idx) const { return m_storage.get()[idx].z(); }
    double varianceR_impl(std::size_t) const { return 0.; }
    double varianceZ_impl(std::size_t) const { return 0.; }

    auto get_impl(std::size_t idx) const { return m_storage.get()[idx]; }

    std::any component_impl(Acts::HashedString key, std::size_t) const {
        using namespace Acts::HashedStringLiteral;
        switch (key) {
            case "TopStripVector"_hash:
            case "BottomStripVector"_hash:
            case "StripCenterDistance"_hash:
            case "TopStripCenterPosition"_hash:
                return Acts::Vector3({0, 0, 0});
            default:
                throw std::invalid_argument("no such component " +
                                            std::to_string(key));
        }
    }

    private:
    std::reference_wrapper<traccc::edm::spacepoint_collection::host> m_storage;
};

class CompareWithActsSeedingTests
    : public ::testing::TestWithParam<
          std::tuple<std::string, std::string, unsigned int>> {};

// This defines the local frame test suite
TEST_P(CompareWithActsSeedingTests, Run) {

    const std::string detector_file = std::get<0>(GetParam());
    const std::string hits_dir = std::get<1>(GetParam());
    const unsigned int event = std::get<2>(GetParam());

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Seeding Config
    traccc::seedfinder_config traccc_config;
    traccc_config.zMin = -1186.f * traccc::unit<float>::mm;
    traccc_config.zMax = 1186.f * traccc::unit<float>::mm;
    traccc_config.cotThetaMax = 7.40627f;
    traccc_config.deltaRMin = 1.f * traccc::unit<float>::mm;
    traccc_config.deltaRMax = 60.f * traccc::unit<float>::mm;
    traccc_config.sigmaScattering = 1.0f;
    traccc_config.deltaZMax = 1000000.f * traccc::unit<float>::mm;

    traccc::spacepoint_grid_config grid_config(traccc_config);

    // Declare algorithms
    traccc::host::details::spacepoint_binning sb{traccc_config, grid_config,
                                                 host_mr};
    traccc::host::details::seed_finding sf{
        traccc_config, traccc::seedfilter_config(), host_mr};

    // Read the hits from the relevant event file
    traccc::edm::spacepoint_collection::host spacepoints_per_event{host_mr};
    traccc::measurement_collection_types::host measurements_per_event{&host_mr};
    traccc::io::read_spacepoints(spacepoints_per_event, measurements_per_event,
                                 event, hits_dir);

    /*--------------------------------
      TRACCC seeding
      --------------------------------*/

    auto internal_spacepoints_per_event =
        sb(vecmem::get_data(spacepoints_per_event));
    auto traccc_seeds = sf(vecmem::get_data(spacepoints_per_event),
                           internal_spacepoints_per_event);

    /*--------------------------------
      ACTS seeding
      --------------------------------*/

    // We need to do some operations on the space points before we can give them
    // to the seeding Config
    Acts::SpacePointContainerConfig spConfig;
    spConfig.useDetailedDoubleMeasurementInfo = false;
    // Options
    Acts::SpacePointContainerOptions spOptions;
    spOptions.beamPos = {traccc_config.beamPos[0], traccc_config.beamPos[1]};

    SpacePointCollector container(spacepoints_per_event);
    SpacePointCollector::ActsSpacePointContainer spContainer(
        spConfig, spOptions, container);
    // The seeding will then iterate on spContainer, that is on space point
    // proxies This also means we will create seed of proxies of space points

    // Define some types
    using spacepoint_t =
        SpacePointCollector::ActsSpacePointContainer::SpacePointProxyType;
    using grid_t = Acts::CylindricalSpacePointGrid<spacepoint_t>;
    using binfinder_t = Acts::GridBinFinder<grid_t::DIM>;
    using binnedgroup_t = Acts::CylindricalBinnedGroup<spacepoint_t>;
    using seedfinderconfig_t = Acts::SeedFinderConfig<spacepoint_t>;
    using seedfinder_t = Acts::SeedFinder<spacepoint_t, grid_t>;
    using seedfilter_t = Acts::SeedFilter<spacepoint_t>;
    using seed_t = Acts::Seed<spacepoint_t, 3ul>;

    // Start creating Seed filter object
    Acts::SeedFilterConfig sfconf;
    // there are a lot more variables here tbh

    // We also need some atlas-specific cut
    Acts::ATLASCuts<spacepoint_t> atlasCuts;

    // Start creating the seed finder object. It needs a Config option
    seedfinderconfig_t acts_config;
    acts_config.seedFilter =
        std::make_shared<seedfilter_t>(sfconf.toInternalUnits(), &atlasCuts);
    // Phi range go from -pi to +pi
    acts_config.phiMin = traccc_config.phiMin;
    acts_config.phiMax = traccc_config.phiMax;
    acts_config.zMin = traccc_config.zMin;
    acts_config.zMax = traccc_config.zMax;
    acts_config.rMin = traccc_config.rMin;
    acts_config.rMax = traccc_config.rMax;

    // zBinEdges is used to understand the radius range of a middle space point
    // candidate This property must be the same as the grid_config.zBinEdges
    // however, it is only used if config.useVariableMiddleSPRange is set to
    // false and if config.rRangeMiddleSP is not empty! Same value as in the
    // seed finder config
    acts_config.zBinEdges = {};

    acts_config.rMinMiddle = 0.f;
    acts_config.rMaxMiddle = std::numeric_limits<float>::max();
    acts_config.useVariableMiddleSPRange = false;
    acts_config.rRangeMiddleSP = {};
    acts_config.deltaRMiddleMinSPRange = 10.f;  // mm
    acts_config.deltaRMiddleMaxSPRange = 10.f;  // mm
    acts_config.deltaRMin = traccc_config.deltaRMin;
    acts_config.deltaRMax = traccc_config.deltaRMax;
    acts_config.deltaRMinTopSP = traccc_config.deltaRMin;
    acts_config.deltaRMaxTopSP = traccc_config.deltaRMax;
    acts_config.deltaRMinBottomSP = traccc_config.deltaRMin;
    acts_config.deltaRMaxBottomSP = traccc_config.deltaRMax;
    acts_config.cotThetaMax = traccc_config.cotThetaMax;
    acts_config.collisionRegionMin = traccc_config.collisionRegionMin;
    acts_config.collisionRegionMax = traccc_config.collisionRegionMax;
    acts_config.maxSeedsPerSpM = traccc_config.maxSeedsPerSpM;
    acts_config.minPt = traccc_config.minPt;
    acts_config.sigmaScattering = traccc_config.sigmaScattering;
    acts_config.maxPtScattering = traccc_config.maxPtScattering;
    acts_config.impactMax = traccc_config.impactMax;
    acts_config.sigmaError = traccc_config.sigmaError;
    acts_config.useDetailedDoubleMeasurementInfo = false;
    // there are other variables here actualy ...
    acts_config = acts_config.toInternalUnits().calculateDerivedQuantities();

    // We create also a Seed Finder Option object
    Acts::SeedFinderOptions acts_options;
    acts_options.bFieldInZ = traccc_config.bFieldInZ;
    acts_options.beamPos[0] = traccc_config.beamPos[0];
    acts_options.beamPos[1] = traccc_config.beamPos[1];
    acts_options =
        acts_options.toInternalUnits().calculateDerivedQuantities(acts_config);

    // Create the grid. This is using a CylindricalSpacePointGrid which is
    // a 3D grid (phi, z, radius). We need to pass a config and an option
    // objects
    Acts::CylindricalSpacePointGridConfig gridConf;
    gridConf.minPt = traccc_config.minPt;
    gridConf.rMax = traccc_config.rMax;
    gridConf.rMin = 0;
    gridConf.zMax = traccc_config.zMax;
    gridConf.zMin = traccc_config.zMin;
    gridConf.deltaRMax = traccc_config.deltaRMax;
    gridConf.cotThetaMax = traccc_config.cotThetaMax;
    gridConf.impactMax = traccc_config.impactMax;
    gridConf.phiMin = traccc_config.phiMin;
    gridConf.phiMax = traccc_config.phiMax;
    // The Bin Ednges allow the user to define a variable binnin in the z or
    // radius azis. If no value is provided, the values of zMin/zMax and
    // rMin/rMax will be used For the phi axis the number of binnings are
    // compute inside the createGrid function
    gridConf.zBinEdges = acts_config.zBinEdges;
    gridConf.rBinEdges = {};
    gridConf.phiBinDeflectionCoverage = traccc_config.phiBinDeflectionCoverage;

    // The b-field is used for defining the number of phi bins in the grid
    // If the b-field is 0, the phi axis of the grid will be automatically set
    // to 100
    Acts::CylindricalSpacePointGridOptions gridOpts;
    gridOpts.bFieldInZ = traccc_config.bFieldInZ;

    grid_t grid =
        Acts::CylindricalSpacePointGridCreator::createGrid<spacepoint_t>(
            gridConf.toInternalUnits(), gridOpts.toInternalUnits());

    // We fill the grid with the space points
    Acts::CylindricalSpacePointGridCreator::fillGrid<spacepoint_t>(
        acts_config, acts_options, grid, spContainer);

    // Perform some checks on the grid definition and its axes.
    // It looks like traccc is usind a 2D grid, while ACTS uses 3D
    // however, the first two axes should be the same: i.e. phi and z
    EXPECT_EQ(grid.numLocalBins().size(), 3ul);

    // Get the traccc axes:
    //  0 -> phi
    //  1 -> z
    detray::axis2::circular axis0 = internal_spacepoints_per_event.axis_p0();
    detray::axis2::regular axis1 = internal_spacepoints_per_event.axis_p1();

    const auto& gridAxes = grid.axes();
    auto acts_axis0 = gridAxes[0];
    auto acts_axis1 = gridAxes[1];

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

    // Define the Bin Finders, these search for neighbours bins
    // The bin finders need instructions to let them know how many neighbour
    // bins we want to retrieve. This is done by providing some inputs to the
    // constructors one value for each axis (provided in the same order of the
    // axes definitions)
    // - if input is vector< pair<int, int> > we need to specify a pair for
    // every bin in the axis
    // - if input is a pair<int, int> these same values are used for all bins in
    // the axis
    // - if input is a int the same values are used for all bins in the axis
    // the constructor can accept a mixture of all of these types
    int numPhiNeighbors = 1;
    int numRadiusNeighbors = 0;
    std::vector<std::pair<int, int>> zBinNeighborsTop{};
    std::vector<std::pair<int, int>> zBinNeighborsBottom{};

    // grid is phi, z, radius
    binfinder_t bottomBinFinder(numPhiNeighbors, zBinNeighborsBottom,
                                numRadiusNeighbors);
    binfinder_t topBinFinder(numPhiNeighbors, zBinNeighborsTop,
                             numRadiusNeighbors);

    // Compute radius Range for the middle space point candidate
    // we rely on the fact the grid is storing the proxies
    // with a sorting in the radius
    float minRange = std::numeric_limits<float>::max();
    float maxRange = std::numeric_limits<float>::lowest();
    for (const auto& coll : grid) {
        if (coll.empty()) {
            continue;
        }
        const auto* firstEl = coll.front();
        const auto* lastEl = coll.back();
        minRange = std::min(firstEl->radius(), minRange);
        maxRange = std::max(lastEl->radius(), maxRange);
    }

    const Acts::Range1D<float> rMiddleSPRange(
        std::floor(minRange / 2) * 2 + acts_config.deltaRMiddleMinSPRange,
        std::floor(maxRange / 2) * 2 - acts_config.deltaRMiddleMaxSPRange);

    // Create Grouping
    // Navigation objects instructs the grouping on how to iterate on the grid
    // This is a collection of vectors of std::size_t, one entry per grid axes
    // The bin group will take the ownership of the grid! Be aware of that.
    std::array<std::vector<std::size_t>, grid_t::DIM> navigation{};
    binnedgroup_t spGroup(std::move(grid), bottomBinFinder, topBinFinder,
                          std::move(navigation));

    // Create the seed filter object
    seedfinder_t a(acts_config);

    // We define the state and the seed container
    std::vector<seed_t> acts_seeds;
    seedfinder_t::SeedingState state;
    state.spacePointMutableData.resize(spContainer.size());

    // Run the seeding
    for (const auto [bottom, middle, top] : spGroup) {
        a.createSeedsForGroup(acts_options, state, spGroup.grid(), acts_seeds,
                              bottom, middle, top, rMiddleSPRange);
    }

    // We have created seeds of proxies to space point at this point
    // From a proxy we can retrieve the original space point object with
    // the externalSpacePoint() method

    // Count the number of matching seeds
    std::size_t n_matched_acts_seeds = 0u;
    for (traccc::edm::seed_collection::host::size_type i = 0u;
         i < traccc_seeds.size(); ++i) {
        const auto traccc_seed = traccc_seeds.at(i);
        // Try to find the same Acts seed.
        auto it = std::find_if(
            acts_seeds.begin(), acts_seeds.end(), [&](const auto& acts_seed) {
                return (
                    (traccc_seed.bottom_index() ==
                     acts_seed.sp()[0]->index()) &&
                    (traccc_seed.middle_index() ==
                     acts_seed.sp()[1]->index()) &&
                    (traccc_seed.top_index() == acts_seed.sp()[2]->index()));
            });

        if (it != acts_seeds.end()) {
            ++n_matched_acts_seeds;
        }
    }

    // Ensure that ACTS and traccc give the same result
    // @TODO Uncomment the line below once acts-project/acts#2132 is merged
    // EXPECT_EQ(seeds.size(), seedVector.size());
    EXPECT_NEAR(static_cast<double>(traccc_seeds.size()),
                static_cast<double>(n_matched_acts_seeds),
                static_cast<double>(traccc_seeds.size()) * 0.004);
    EXPECT_GT(static_cast<double>(n_matched_acts_seeds) /
                  static_cast<double>(traccc_seeds.size()),
              0.995);
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
