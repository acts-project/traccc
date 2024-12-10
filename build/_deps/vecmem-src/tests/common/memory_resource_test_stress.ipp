/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/vector.hpp"

// System include(s).
#include <vector>

/// Test that the memory resource would behave correctly with a large number
/// of allocations/de-allocations.
TEST_P(memory_resource_test_stress, stress_test) {

    // Repeat the allocations multiple times.
    for (int i = 0; i < 100; ++i) {

        // Create an object that would hold on to the allocated memory
        // "for one iteration".
        std::vector<vecmem::vector<int> > vectors;

        // Fill a random number of vectors.
        const int n_vectors = std::rand() % 100;
        for (int j = 0; j < n_vectors; ++j) {

            // Fill them with a random number of "constant" elements.
            vectors.emplace_back(GetParam());
            const int n_elements = std::rand() % 100;
            for (int k = 0; k < n_elements; ++k) {
                vectors.back().push_back(j);
            }
        }

        // Check that all vectors have the intended content after all of this.
        for (int j = 0; j < n_vectors; ++j) {
            for (int value : vectors.at(static_cast<std::size_t>(j))) {
                EXPECT_EQ(value, j);
            }
        }
    }
}
