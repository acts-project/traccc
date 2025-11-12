/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

// Detray include(s).
#include <detray/definitions/algebra.hpp>
#include <detray/geometry/barcode.hpp>

// VecMem include(s).
#include <vecmem/edm/container.hpp>

// System include(s).
#include <array>
#include <compare>

namespace traccc::edm {

/// Interface for the @c traccc::edm::measurement_collection type.
///
/// It provides the API that users would interact with, while using the
/// columns/arrays of the SoA containers, or the variables of the AoS proxies
/// created on top of the SoA containers.
///
template <typename BASE>
class measurement : public BASE {

    public:
    /// @name Functions inherited from the base class
    /// @{

    /// Inherit the base class's constructor(s)
    using BASE::BASE;
    /// Inherit the base class's assignment operator(s).
    using BASE::operator=;

    /// @}

    /// @name Measurement Information
    /// @{

    /// Local position of the measurement (non-const)
    ///
    /// @return A (non-const) vector of 1D/2D points
    ///
    TRACCC_HOST_DEVICE
    auto& local_position() { return BASE::template get<0>(); }
    /// Local position of the measurement (const)
    ///
    /// @return A (const) vector of 1D/2D points
    ///
    TRACCC_HOST_DEVICE
    const auto& local_position() const { return BASE::template get<0>(); }

    /// Variance of the local position of the measurement (non-const)
    ///
    /// @return A (non-const) vector of 1D/2D variances
    ///
    TRACCC_HOST_DEVICE
    auto& local_variance() { return BASE::template get<1>(); }
    /// Variance of the local position of the measurement (const)
    ///
    /// @return A (const) vector of 1D/2D variances
    ///
    TRACCC_HOST_DEVICE
    const auto& local_variance() const { return BASE::template get<1>(); }

    /// Dimensionality of the measurement (non-const)
    ///
    /// @return A (non-const) vector of unsigned integers
    ///
    TRACCC_HOST_DEVICE
    auto& dimensions() { return BASE::template get<2>(); }
    /// Dimensionality of the measurement (const)
    ///
    /// @return A (const) vector of unsigned integers
    ///
    TRACCC_HOST_DEVICE
    const auto& dimensions() const { return BASE::template get<2>(); }

    /// Time assigned to the measurement (non-const)
    ///
    /// @return A (non-const) vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    auto& time() { return BASE::template get<3>(); }
    /// Time assigned to the measurement (const)
    ///
    /// @return A (const) vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    const auto& time() const { return BASE::template get<3>(); }

    /// Diameter of the measurement (non-const)
    ///
    /// @return A (non-const) vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    auto& diameter() { return BASE::template get<4>(); }
    /// Diameter of the measurement (const)
    ///
    /// @return A (const) vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    const auto& diameter() const { return BASE::template get<4>(); }

    /// Unique measurement identifier (non-const)
    ///
    /// @return A (non-const) vector of unsigned integer values
    ///
    TRACCC_HOST_DEVICE
    auto& identifier() { return BASE::template get<5>(); }
    /// Unique measurement identifier (const)
    ///
    /// @return A (const) vector of unsigned integer values
    ///
    TRACCC_HOST_DEVICE
    const auto& identifier() const { return BASE::template get<5>(); }

    /// Identifier of the tracking sufrace of the measurement (non-const)
    ///
    /// @return A (non-const) vector of geometry barcodes
    ///
    TRACCC_HOST_DEVICE
    auto& surface_link() { return BASE::template get<6>(); }
    /// Identifier of the tracking sufrace of the measurement (const)
    ///
    /// @return A (const) vector of geometry barcodes
    ///
    TRACCC_HOST_DEVICE
    const auto& surface_link() const { return BASE::template get<6>(); }

    /// Subspace of the measurement (non-const)
    ///
    /// @return A (non-const) vector of subspace objects
    ///
    TRACCC_HOST_DEVICE
    auto& subspace() { return BASE::template get<7>(); }
    /// Subspace of the measurement (const)
    ///
    /// @return A (const) vector of subspace objects
    ///
    TRACCC_HOST_DEVICE
    const auto& subspace() const { return BASE::template get<7>(); }

    /// Index of the cluster that the measurement was created from (non-const)
    ///
    /// @return A (non-const) vector of unsigned integers
    ///
    TRACCC_HOST_DEVICE
    auto& cluster_index() { return BASE::template get<8>(); }
    /// Index of the cluster that the measurement was created from (const)
    ///
    /// @return A (const) vector of unsigned integers
    ///
    TRACCC_HOST_DEVICE
    const auto& cluster_index() const { return BASE::template get<8>(); }

    /// @}

    /// @name Utility functions
    /// @{

    /// Equality operator
    ///
    /// @note This function must only be used on proxy objects, not on
    ///       containers!
    ///
    /// @param[in] other The object to compare with
    /// @return @c true if the objects are equal, @c false otherwise
    ///
    template <typename T>
    TRACCC_HOST_DEVICE bool operator==(const measurement<T>& other) const;

    /// Comparison operator
    ///
    /// @note This function must only be used on proxy objects, not on
    ///       containers!
    ///
    /// @param[in] other The object to compare with
    /// @return A weak ordering object, describing the relation between the
    ///         two objects
    ///
    template <typename T>
    TRACCC_HOST_DEVICE std::partial_ordering operator<=>(
        const measurement<T>& other) const;

    /// @}

};  // class measurement

/// SoA container of measurements
template <detray::concepts::algebra ALGEBRA>
using measurement_collection =
    vecmem::edm::container<measurement,
                           // local_position
                           vecmem::edm::type::vector<detray::dpoint2D<ALGEBRA>>,
                           // local_variance
                           vecmem::edm::type::vector<detray::dpoint2D<ALGEBRA>>,
                           // dimensions
                           vecmem::edm::type::vector<unsigned int>,
                           // time
                           vecmem::edm::type::vector<detray::dscalar<ALGEBRA>>,
                           // diameter
                           vecmem::edm::type::vector<detray::dscalar<ALGEBRA>>,
                           // identifier
                           vecmem::edm::type::vector<unsigned int>,
                           // surface_link
                           vecmem::edm::type::vector<detray::geometry::barcode>,
                           // subspace
                           vecmem::edm::type::vector<std::array<
                               detray::dsize_type<default_algebra>, 2u>>,
                           // cluster_index
                           vecmem::edm::type::vector<unsigned int>>;

}  // namespace traccc::edm

// Include the implementation.
#include "traccc/edm/impl/measurement_collection.ipp"
