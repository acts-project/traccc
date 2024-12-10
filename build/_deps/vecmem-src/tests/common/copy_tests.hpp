/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <tuple>

/// Test case for @c vecmem::copy specializations
///
/// It tests the copy of different data types, using the provided copy objects
/// and memory resources.
///
class copy_tests
    : public testing::TestWithParam<
          std::tuple<vecmem::copy*&, vecmem::copy*&, vecmem::memory_resource*&,
                     vecmem::memory_resource*&>> {

public:
    /// Set up the test fixture.
    void SetUp() override;

protected:
    /// Access the "main" copy object
    vecmem::copy& main_copy();
    /// Access the "host" copy object
    vecmem::copy& host_copy();
    /// Access the "main" / "device" memory resource
    vecmem::memory_resource& main_mr();
    /// Access the "host" memory resource
    vecmem::memory_resource& host_mr();
    /// Access the "host" memory resource
    vecmem::memory_resource* host_mr_ptr();

    /// Access the reference 1D data (non-const)
    vecmem::vector<int>& ref();
    /// Access the reference 1D data (const)
    const vecmem::vector<int>& cref() const;

    /// Access the reference jagged data (non-const)
    vecmem::jagged_vector<int>& jagged_ref();
    /// Access the reference jagged data (const)
    const vecmem::jagged_vector<int>& jagged_cref() const;

private:
    /// 1D reference data for the tests.
    vecmem::vector<int> m_ref;
    /// Jagged reference data for the tests.
    vecmem::jagged_vector<int> m_jagged_ref;
};

// Include the implementation.
#include "copy_tests.ipp"
