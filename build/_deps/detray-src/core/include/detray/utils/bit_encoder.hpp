
/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/definitions/detail/qualifiers.hpp"

// System include(s)
#include <cstdint>
#include <type_traits>

namespace detray::detail {

/// @brief Sets masked bits to a given value.
///
/// Given a mask and an input value, the corresponding bits are set on a target
/// value. Used e.g. in the surface barcode.
///
/// @see
/// https://github.com/acts-project/acts/blob/main/Core/include/Acts/Geometry/GeometryIdentifier.hpp
template <typename value_t = std::uint64_t>
class bit_encoder {

    public:
    bit_encoder() = delete;

    /// Check wether the set @param v encodes valid values according to the
    /// given masks (@tparam head and @tparam tail)
    template <value_t... masks>
    DETRAY_HOST_DEVICE static constexpr bool is_invalid(value_t v) noexcept {
        // All bits set to one in the range of a given mask defined as invalid
        return (((v & masks) == masks) || ...);
    }

    /// @returns the masked bits from the encoded value as value of the same
    /// type.
    template <value_t mask>
    requires(mask != static_cast<value_t>(0)) DETRAY_HOST_DEVICE
        static constexpr value_t get_bits(const value_t v) noexcept {
        // Use integral constant to enforce compile time evaluation of shift
        return (v & mask) >> extract_shift(mask);
    }

    /// Set the masked bits to id in the encoded value.
    template <value_t mask>
    requires(mask != static_cast<value_t>(0)) DETRAY_HOST_DEVICE
        static constexpr void set_bits(value_t& v, const value_t id) noexcept {
        // Use integral constant to enforce compile time evaluation of shift
        v = (v & ~mask) | ((id << extract_shift(mask)) & mask);
    }

    private:
    /// Extract the bit shift necessary to access the masked values.
    ///
    /// @note undefined behaviour for mask == 0 which we should not have.
    DETRAY_HOST_DEVICE
    static consteval int extract_shift(value_t mask) noexcept {
        return __builtin_ctzll(mask);
    }
};

}  // namespace detray::detail
