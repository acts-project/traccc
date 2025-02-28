/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

// Detray include(s).
#include <detray/geometry/barcode.hpp>

// VecMem include(s).
#include <vecmem/edm/container.hpp>

namespace traccc {

/// Interface for the @c traccc::silicon_detector_description class.
///
/// It provides the API that users would interact with, while using the
/// columns/arrays defined in @c traccc::silicon_detector_description.
///
template <typename BASE>
class silicon_detector_description_interface : public BASE {

    public:
    /// Inherit the base class's constructor(s)
    using BASE::BASE;

    /// @name Detector geometry information
    /// @{

    /// The identifier of the detector module's surface (non-const)
    ///
    /// Can be used to look up the module in a @c detray::detector object.
    ///
    /// @return A (non-const) vector of @c detray::geometry::barcode objects
    ///
    TRACCC_HOST_DEVICE
    auto& geometry_id() { return BASE::template get<0>(); }
    /// The identifier of the detector module's surface (const)
    ///
    /// Can be used to look up the module in a @c detray::detector object.
    ///
    /// @return A (const) vector of @c detray::geometry::barcode objects
    ///
    TRACCC_HOST_DEVICE
    const auto& geometry_id() const { return BASE::template get<0>(); }

    /// @}

    /// @name Detector module information
    /// @{

    /// Acts geometry identifier for the detector module (non-const)
    ///
    /// It is the "Acts geometry ID" for the module, as used in the simulation
    /// files that we use.
    ///
    /// @return A (non-const) vector of @c traccc::geometry_id values
    ///
    TRACCC_HOST_DEVICE
    auto& acts_geometry_id() { return BASE::template get<1>(); }
    /// Acts geometry identifier for the detector module (const)
    ///
    /// It is the "Acts geometry ID" for the module, as used in the simulation
    /// files that we use.
    ///
    /// @return A (const) vector of @c traccc::geometry_id values
    ///
    TRACCC_HOST_DEVICE
    const auto& acts_geometry_id() const { return BASE::template get<1>(); }

    /// Signal threshold for detection elements (non-const)
    ///
    /// It controls which elements (pixels and strips) are considered during
    /// clusterization.
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& threshold() { return BASE::template get<2>(); }
    /// Signal threshold for detection elements (const)
    ///
    /// It controls which elements (pixels and strips) are considered during
    /// clusterization.
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& threshold() const { return BASE::template get<2>(); }

    /// Reference for local position calculation in X direction (non-const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the X axis with the formula:
    /// @f$pos_x = reference_x + pitch_x * index_x@f$
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& reference_x() { return BASE::template get<3>(); }
    /// Reference for local position calculation in X direction (const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the X axis with the formula:
    /// @f$pos_x = reference_x + pitch_x * index_x@f$
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& reference_x() const { return BASE::template get<3>(); }

    /// Reference for local position calculation in Y direction (non-const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the Y axis with the formula:
    /// @f$pos_y = reference_y + pitch_y * index_y@f$
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& reference_y() { return BASE::template get<4>(); }
    /// Reference for local position calculation in Y direction (const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the Y axis with the formula:
    /// @f$pos_y = reference_y + pitch_y * index_y@f$
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& reference_y() const { return BASE::template get<4>(); }

    /// Pitch for local position calculation in X direction (non-const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the X axis with the formula:
    /// @f$pos_x = reference_x + pitch_x * index_x@f$
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& pitch_x() { return BASE::template get<5>(); }
    /// Pitch for local position calculation in X direction (const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the X axis with the formula:
    /// @f$pos_x = reference_x + pitch_x * index_x@f$
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& pitch_x() const { return BASE::template get<5>(); }

    /// Pitch for local position calculation in Y direction (non-const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the Y axis with the formula:
    /// @f$pos_y = reference_y + pitch_y * index_y@f$
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& pitch_y() { return BASE::template get<6>(); }
    /// Pitch for local position calculation in Y direction (const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the Y axis with the formula:
    /// @f$pos_y = reference_y + pitch_y * index_y@f$
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& pitch_y() const { return BASE::template get<6>(); }

    /// The dimensionality (1D/2D) of the detector module (non-const)
    ///
    /// @return A (non-const) vector of @c char objects
    ///
    TRACCC_HOST_DEVICE
    auto& dimensions() { return BASE::template get<7>(); }
    /// The dimensionality (1D/2D) of the detector module (const)
    ///
    /// @return A (const) vector of @c char objects
    ///
    TRACCC_HOST_DEVICE
    const auto& dimensions() const { return BASE::template get<7>(); }

    /// @}

};  // class silicon_detector_description_interface

/// SoA container describing a silicon detector
using silicon_detector_description = vecmem::edm::container<
    silicon_detector_description_interface,
    vecmem::edm::type::vector<detray::geometry::barcode>,
    vecmem::edm::type::vector<geometry_id>, vecmem::edm::type::vector<scalar>,
    vecmem::edm::type::vector<scalar>, vecmem::edm::type::vector<scalar>,
    vecmem::edm::type::vector<scalar>, vecmem::edm::type::vector<scalar>,
    vecmem::edm::type::vector<unsigned char> >;

}  // namespace traccc
