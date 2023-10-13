/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/track_parametrization.hpp"

// System include(s).
#include <array>

namespace traccc {

template <typename algebra_t, typename algebra_t::size_type kFullSize,
          typename algebra_t::size_type kSize = 2u>
struct subspace {
    static_assert(1u <= kSize, "Subspace size must be at least 1");
    static_assert(kSize <= kFullSize,
                  "Subspace can only be as large as the full space");

    public:
    // Type declarations
    using matrix_operator = typename algebra_t::matrix_actor;
    using size_type = typename matrix_operator::size_ty;
    template <size_type ROWS, size_type COLS>
    using matrix_type =
        typename matrix_operator::template matrix_type<ROWS, COLS>;

    using subspace_vector = matrix_type<kSize, 1u>;
    using fullspace_vector = matrix_type<kFullSize, 1u>;
    using projection_matrix = matrix_type<kSize, kFullSize>;
    using expansion_matrix = matrix_type<kFullSize, kSize>;

    static const size_type size = kSize;
    static const size_type fullSize = kFullSize;

    /// Construct from a container of axis indices.
    ///
    /// @param indices Unique, ordered indices
    TRACCC_HOST_DEVICE
    constexpr subspace(const std::array<size_type, kSize>& indices) {
        for (size_type i = 0u; i < kSize; ++i) {
            assert((indices[i] < kFullSize) and
                   "Axis indices must be within the full space");
            if (0u < i) {
                assert((indices[i - 1u] < indices[i]) and
                       "Axis indices must be unique and ordered");
            }
        }
        for (size_type i = 0; i < kSize; ++i) {
            m_axes[i] = static_cast<size_type>(indices[i]);
        }
    }

    /// Axis indices that comprise the subspace.
    ///
    /// The specific container and index type should be considered an
    /// implementation detail. Users should treat the return type as a generic
    /// container whose elements are convertible to `size_t`.
    TRACCC_HOST_DEVICE
    constexpr const std::array<size_type, kSize>& get_indices() const {
        return m_axes;
    }

    /// Function that sets the m_axes
    TRACCC_HOST_DEVICE
    void set_indices(const std::array<size_type, kSize>& indices) {
        m_axes = indices;
    }

    /// Projection matrix that maps from the full space into the subspace.
    template <size_type D>
    TRACCC_HOST_DEVICE auto projector() const -> matrix_type<D, kFullSize> {
        static_assert(D <= kSize,
                      "The dimension of projection should be smaller than "
                      "subspace dimension");

        matrix_type<D, kFullSize> proj =
            matrix_operator().template zero<D, kFullSize>();

        for (size_type i = 0u; i < D; ++i) {
            getter::element(proj, i, m_axes[i]) = 1;
        }
        return proj;
    }

    /// Expansion matrix that maps from the subspace into the full space.
    template <size_type D>
    TRACCC_HOST_DEVICE auto expander() const -> matrix_type<kFullSize, D> {
        static_assert(D <= kSize,
                      "The dimension of projection should be smaller than "
                      "subspace dimension");

        matrix_type<kFullSize, D> expn =
            matrix_operator().template zero<kFullSize, D>();

        for (size_type i = 0u; i < kSize; ++i) {
            getter::element(expn, m_axes[i], i) = 1;
        }
        return expn;
    }

    private:
    std::array<size_type, kSize> m_axes;
};

}  // namespace traccc