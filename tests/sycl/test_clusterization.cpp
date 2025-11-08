/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "tests/cca_test.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/sycl/clusterization/clusterization_algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/sycl/shared_memory_resource.hpp>
#include <vecmem/utils/sycl/copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

TEST(SYCLClustering, SingleModule) {

    // Creating SYCL queue object
    vecmem::sycl::queue_wrapper vecmem_queue;
    traccc::sycl::queue_wrapper traccc_queue{vecmem_queue.queue()};
    std::cout << "Running on device: " << vecmem_queue.device_name() << "\n";

    // Memory resource used by the EDM.
    vecmem::sycl::shared_memory_resource shared_mr{vecmem_queue};
    traccc::memory_resource mr{shared_mr};

    // Copy object
    vecmem::sycl::copy copy{vecmem_queue};

    // Create cell collection
    traccc::edm::silicon_cell_collection::host cells{shared_mr};
    cells.reserve(8u);
    cells.push_back({1u, 2u, 1.f, 0.f, 0u});
    cells.push_back({2u, 2u, 1.f, 0.f, 0u});
    cells.push_back({3u, 2u, 1.f, 0.f, 0u});
    cells.push_back({6u, 4u, 1.f, 0.f, 0u});
    cells.push_back({5u, 5u, 1.f, 0.f, 0u});
    cells.push_back({6u, 5u, 1.f, 0.f, 0u});
    cells.push_back({7u, 5u, 1.f, 0.f, 0u});
    cells.push_back({6u, 6u, 1.f, 0.f, 0u});

    // Create a dummy detector description.
    traccc::silicon_detector_description::host dd{shared_mr};
    dd.resize(1u);
    dd.reference_x()[0] = 0.f;
    dd.reference_y()[0] = 0.f;
    dd.pitch_x()[0] = 1.f;
    dd.pitch_y()[0] = 1.f;
    dd.dimensions()[0] = 2;
    dd.geometry_id()[0] = detray::geometry::barcode{0u};

    // Run Clusterization
    traccc::sycl::clusterization_algorithm ca_sycl(mr, copy, traccc_queue,
                                                   default_ccl_test_config());

    auto measurements_buffer =
        ca_sycl(vecmem::get_data(cells), vecmem::get_data(dd));

    edm::measurement_collection<default_algebra>::device measurements(
        measurements_buffer);

    // Check the results
    EXPECT_EQ(copy.get_size(measurements_buffer), 2u);

    edm::measurement_collection<default_algebra>::host references{shared_mr};
    references.push_back({{2.5f, 2.5f},
                          {0.75f, 0.0833333f},
                          2u,
                          0.f,
                          0.f,
                          0u,
                          detray::geometry::barcode{0u},
                          {1u, 1u},
                          0u});
    references.push_back({{6.5f, 5.5f},
                          {0.483333f, 0.483333f},
                          2u,
                          0.f,
                          0.f,
                          0u,
                          detray::geometry::barcode{0u},
                          {1u, 1u},
                          1u});

    for (unsigned int i = 0; i < measurements.size(); ++i) {
        const auto test = measurements.at(i);
        // 0.01 % uncertainty
        auto iso = traccc::details::is_same_object<edm::measurement_collection<
            default_algebra>::const_device::object_type>(test, 0.0001f);
        bool matched = false;

        for (std::size_t j = 0; j < references.size(); ++j) {
            const auto ref = references.at(j);
            if (iso(ref)) {
                matched = true;
                break;
            }
        }

        ASSERT_TRUE(matched);
    }
}
