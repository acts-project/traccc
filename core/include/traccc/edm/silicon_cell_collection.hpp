/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/edm/container.hpp>

// System include(s).
#include <type_traits>

namespace traccc::edm {

/// Interface for the @c traccc::edm::silicon_cell_collection class.
///
/// It provides the API that users would interact with, while using the
/// columns/arrays of the SoA containers, or the variables of the AoS proxies
/// created on top of the SoA containers.
///
template <typename BASE>
class silicon_cell : public BASE {

    public:
    /// @name Constructors
    /// @{

    /// Inherit the base class's constructor(s)
    using BASE::BASE;
    /// Use a default copy constructor
    silicon_cell(const silicon_cell& other) = default;
    /// Use a default move constructor
    silicon_cell(silicon_cell&& other) = default;

    /// @}

    /// @name Cell Information
    /// @{

    /// The first channel identifier / index of the cell / strip (non-const)
    ///
    /// @return A (non-const) vector of @c traccc::channel_id values
    ///
    TRACCC_HOST_DEVICE
    auto& channel0() { return BASE::template get<0>(); }
    /// The first channel identifier / index of the cell / strip (const)
    ///
    /// @return A (const) vector of @c traccc::channel_id values
    ///
    TRACCC_HOST_DEVICE
    const auto& channel0() const { return BASE::template get<0>(); }

    /// The second channel identifier / index of the cell / strip (non-const)
    ///
    /// @return A (non-const) vector of @c traccc::channel_id values
    ///
    TRACCC_HOST_DEVICE
    auto& channel1() { return BASE::template get<1>(); }
    /// The second channel identifier / index of the cell / strip (const)
    ///
    /// @return A (const) vector of @c traccc::channel_id values
    ///
    TRACCC_HOST_DEVICE
    const auto& channel1() const { return BASE::template get<1>(); }

    /// The "activation" of the cell / strip (non-const)
    ///
    /// @return A (non-const) vector of @c traccc::scalar values
    ///
    TRACCC_HOST_DEVICE
    auto& activation() { return BASE::template get<2>(); }
    /// The "activation" of the cell / strip (const)
    ///
    /// @return A (const) vector of @c traccc::scalar values
    ///
    TRACCC_HOST_DEVICE
    const auto& activation() const { return BASE::template get<2>(); }

    /// The time associated with the cell / strip (non-const)
    ///
    /// @return A (non-const) vector of @c traccc::scalar values
    ///
    TRACCC_HOST_DEVICE
    auto& time() { return BASE::template get<3>(); }
    /// The time associated with the cell / strip (const)
    ///
    /// @return A (const) vector of @c traccc::scalar values
    ///
    TRACCC_HOST_DEVICE
    const auto& time() const { return BASE::template get<3>(); }

    /// The index of the module that this cell belongs to (non-const)
    ///
    /// Used to look up the module in @c traccc::silicon_detector_description.
    ///
    /// @return A (non-const) vector of <tt>unsigned int</tt> values
    ///
    TRACCC_HOST_DEVICE
    auto& module_index() { return BASE::template get<4>(); }
    /// The index of the module that this cell belongs to (const)
    ///
    /// Used to look up the module in @c traccc::silicon_detector_description.
    ///
    /// @return A (const) vector of <tt>unsigned int</tt> values
    ///
    TRACCC_HOST_DEVICE
    const auto& module_index() const { return BASE::template get<4>(); }

    /// @}

    /// @name Utility functions
    /// @{

    /// Assignment operator
    ///
    /// @param[in] other The object to assign from
    /// @return A reference to this object
    ///
    TRACCC_HOST_DEVICE silicon_cell& operator=(const silicon_cell& other) {
        channel0() = other.channel0();
        channel1() = other.channel1();
        activation() = other.activation();
        time() = other.time();
        module_index() = other.module_index();
        return *this;
    }

    /// Assignment operator
    ///
    /// @param[in] other The object to assign from
    /// @return A reference to this object
    ///
    template <typename T,
              std::enable_if_t<!std::is_same_v<BASE, T>, bool> = false>
    TRACCC_HOST_DEVICE silicon_cell& operator=(const silicon_cell<T>& other) {
        channel0() = other.channel0();
        channel1() = other.channel1();
        activation() = other.activation();
        time() = other.time();
        module_index() = other.module_index();
        return *this;
    }

    /// Comparison operator
    ///
    /// @note This function must only be used on proxy objects, not on
    ///       containers!
    ///
    /// @param[in] other The object to compare with
    /// @return @c true if the objects are equal, @c false otherwise
    ///
    template <typename T>
    TRACCC_HOST_DEVICE bool operator==(const silicon_cell<T>& other) const {
        return (channel0() == other.channel0()) &&
               (channel1() == other.channel1()) &&
               (math::fabs(activation() - other.activation()) < 1e-6f) &&
               (math::fabs(time() - other.time()) < 1e-6f) &&
               (module_index() == other.module_index());
    }

    /// Ordering operator
    ///
    /// @note This function must only be used on proxy objects, not on
    ///       containers!
    ///
    /// @param[in] other The object to compare with
    /// @return @c true if this object is less than the other, @c false
    ///         otherwise
    ///
    template <typename T>
    TRACCC_HOST_DEVICE bool operator<(const silicon_cell<T>& other) const {
        if (module_index() != other.module_index()) {
            return (module_index() < other.module_index());
        } else {
            if (channel1() != other.channel1()) {
                return (channel1() < other.channel1());
            } else {
                return (channel0() < other.channel0());
            }
        }
    }

    /// @}

};  // class silicon_cell_collection_interface

/// SoA container describing silicon detector hits
using silicon_cell_collection =
    vecmem::edm::container<silicon_cell, vecmem::edm::type::vector<channel_id>,
                           vecmem::edm::type::vector<channel_id>,
                           vecmem::edm::type::vector<scalar>,
                           vecmem::edm::type::vector<scalar>,
                           vecmem::edm::type::vector<unsigned int> >;

}  // namespace traccc::edm
