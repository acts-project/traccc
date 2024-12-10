/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/container.hpp"
#include "vecmem/utils/types.hpp"

namespace vecmem {
namespace testing {

/// Interface to a "jagged container" used for testing
template <typename BASE>
class jagged_soa_interface : public BASE {

public:
    /// Inherit the base class's constructor(s)
    using BASE::BASE;

    /// Global "count" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& count() { return BASE::template get<0>(); }
    /// Global "count" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& count() const { return BASE::template get<0>(); }

    /// "Measurement" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& measurement() { return BASE::template get<1>(); }
    /// "Measurement" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& measurement() const { return BASE::template get<1>(); }

    /// "Measurements" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& measurements() { return BASE::template get<2>(); }
    /// "Measurements" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& measurements() const { return BASE::template get<2>(); }

    /// Global "average" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& average() { return BASE::template get<3>(); }
    /// Global "average" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& average() const { return BASE::template get<3>(); }

    /// "Indices" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& indices() { return BASE::template get<4>(); }
    /// "Indices" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& indices() const { return BASE::template get<4>(); }

    /// "Index" of something (non-const)
    VECMEM_HOST_AND_DEVICE
    auto& index() { return BASE::template get<5>(); }
    /// "Index" of something (const)
    VECMEM_HOST_AND_DEVICE
    const auto& index() const { return BASE::template get<5>(); }

};  // class jagged_interface

/// "Jagged" container for the tests
using jagged_soa_container =
    edm::container<jagged_soa_interface, edm::type::scalar<int>,
                   edm::type::vector<float>, edm::type::jagged_vector<double>,
                   edm::type::scalar<float>, edm::type::jagged_vector<int>,
                   edm::type::vector<int> >;

}  // namespace testing
}  // namespace vecmem
