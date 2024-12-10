/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/algorithms.hpp"
#include "detray/definitions/detail/indexing.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/grid_axis.hpp"

// System include(s).
#include <cstddef>
#include <iterator>

namespace detray::axis {

/// @brief Helper to tie two bin indices to a range.
/// @note Cannot use dindex_range for signed integer bin indices.
using bin_range = std::array<int, 2>;

/// @brief A regular binning scheme.
///
/// The binning on the input parameter space is regular and therefore only needs
/// the span and the number of bins to match a value to a bin.
template <typename dcontainers = host_container_types,
          typename scalar_t = scalar>
struct regular {

    // Extract container types
    using scalar_type = scalar_t;
    using container_types = dcontainers;
    template <typename T>
    using vector_type = typename dcontainers::template vector_type<T>;
    template <typename T, std::size_t N>
    using array_type = typename dcontainers::template array_type<T, N>;

    static constexpr binning type = binning::e_regular;

    /// Offset into the bin edges container and the number of bins
    dindex m_offset{0};
    dindex m_n_bins{0};

    /// Access to the bin edges
    const vector_type<scalar_type> *m_bin_edges{nullptr};

    /// Default constructor (no concrete memory acces)
    constexpr regular() = default;

    /// Constructor from an index range and bin edges - non-owning
    ///
    /// @param range range of bin boundary entries in an external storage
    /// @param edges lower edges for all bins in an external storage
    DETRAY_HOST_DEVICE
    regular(const dindex_range &range, const vector_type<scalar_type> *edges)
        : m_offset(detray::detail::get<0>(range)),
          m_n_bins{detray::detail::get<1>(range)},
          m_bin_edges(edges) {}

    /// @returns the total number of bins, which for the regular axis is simply
    /// the second entry in the range
    DETRAY_HOST_DEVICE
    dindex nbins() const { return m_n_bins; }

    /// Access function to a single bin from a value v
    ///
    /// @param v is the value for the bin search
    ///
    /// @note the floating point truncation to integer leaves
    /// int(-0.9) = int(0.9) = 0, which wrongfully maps overflow bins onto the
    /// axis range itself.
    /// Therefore, the values are shifted towards positive bin numbers first,
    /// and then shifted back, in order to emulate @c std::floor for this case.
    ///
    /// @returns the corresponding bin index
    DETRAY_HOST_DEVICE
    int bin(const scalar_type v) const {
        return static_cast<int>((v - span()[0]) / bin_width() + 1.f) - 1;
    }

    /// Access function to a range with binned neighborhood
    ///
    /// @note This is an inclusive range
    ///
    /// @param v is the value for the bin search
    /// @param nhood is the neighborhood bin index range (# neighboring bins)
    ///
    /// @returns the corresponding range of bin indices
    DETRAY_HOST_DEVICE
    bin_range range(const scalar_type v,
                    const array_type<dindex, 2> &nhood) const {
        const int ibin{bin(v)};
        const int ibinmin{ibin - static_cast<int>(nhood[0])};
        const int ibinmax{ibin + static_cast<int>(nhood[1])};

        return {ibinmin, ibinmax};
    }

    /// Access function to a range with scalar neighborhood
    ///
    /// @note This is an inclusive range
    ///
    /// @param v is the value for the bin search
    /// @param nhood is the neighborhood value range (range on axis values)
    ///
    /// @returns the corresponding range of bin indices
    DETRAY_HOST_DEVICE
    bin_range range(const scalar_type v,
                    const array_type<scalar_type, 2> &nhood) const {
        return {bin(v - nhood[0]), bin(v + nhood[1])};
    }

    /// @return the bin edges for a given @param ibin
    DETRAY_HOST_DEVICE
    array_type<scalar_type, 2> bin_edges(const dindex ibin) const {
        const scalar_type width{bin_width()};
        const scalar_type lower_edge{span()[0] +
                                     static_cast<scalar_type>(ibin) * width};
        return {lower_edge, lower_edge + width};
    }

    /// @return the values of the edges of all bins - uses dynamic memory
    // TODO: return generator view instead to make it work properly in device
    DETRAY_HOST_DEVICE
    vector_type<scalar_type> bin_edges() const {
        // Output vector has to be constructed, because the edges are
        // calculated on the fly
        vector_type<scalar_type> edges;
        detray::detail::call_reserve(edges, nbins());

        // Calculate bin edges from number of bins and axis span
        const array_type<scalar_type, 2> sp = span();
        const scalar_type step{bin_width()};

        for (dindex ib = 0; ib <= nbins(); ++ib) {
            edges.push_back(sp[0] + static_cast<scalar_type>(ib) * step);
        }

        return edges;
    }

    /// @return the bin width between any two bins.
    DETRAY_HOST_DEVICE
    scalar_type bin_width() const {
        // Get the binning information
        const scalar_type min{(*m_bin_edges)[m_offset]};
        const scalar_type max{(*m_bin_edges)[m_offset + 1]};

        const scalar_type step_size{(max - min) /
                                    static_cast<scalar_type>(nbins())};

        return step_size;
    }

    /// @return the span of the binning (equivalent to the span of the axis:
    /// [min, max) )
    DETRAY_HOST_DEVICE
    array_type<scalar_type, 2> span() const {
        // Get the binning information
        const scalar_type min{(*m_bin_edges)[m_offset]};
        const scalar_type max{(*m_bin_edges)[m_offset + 1]};

        return {min, max};
    }

    /// Equality operator
    ///
    /// @param rhs the axis to compare to
    ///
    /// @note as we cannot guarantee to have the same pointer for the bin edges,
    /// we make a fast comparison of the pointer first, but also allow for a
    /// value based comparison
    ///
    /// @returns whether the two axes are equal
    DETRAY_HOST_DEVICE constexpr bool operator==(const regular &rhs) const {
        return (nbins() == rhs.nbins()) && (span() == rhs.span());
    }
};

/// @brief An irregular binning scheme.
///
/// The bin edges are irregular in the underlying parameter space.
/// Therefore, the correct bin index has to be searched for.
///
/// @note The bin search makes this type comparatively expensive. Only use when
/// absolutely needed.
template <typename dcontainers = host_container_types,
          typename scalar_t = scalar>
struct irregular {

    // Extract container types
    using scalar_type = scalar_t;
    using container_types = dcontainers;
    template <typename T>
    using vector_type = typename dcontainers::template vector_type<T>;
    template <typename T, std::size_t N>
    using array_type = typename dcontainers::template array_type<T, N>;
    using index_type = typename std::iterator_traits<
        typename vector_type<scalar_type>::iterator>::difference_type;

    static constexpr binning type = binning::e_irregular;

    /// Offset into the bin edges container and the number of bins
    dindex m_offset{0};
    dindex m_n_bins{0};

    /// Access to the bin edges
    const vector_type<scalar_type> *m_bin_edges{nullptr};

    /// Default constructor (no concrete memory access)
    constexpr irregular() = default;

    /// Constructor from an index range and bin edges - non-owning
    ///
    /// @param range range of bin boundary entries in an external storage
    /// @param edges lower edges for all bins in an external storage
    DETRAY_HOST_DEVICE
    irregular(const dindex_range &range, const vector_type<scalar_type> *edges)
        : m_offset(detray::detail::get<0>(range)),
          m_n_bins{detray::detail::get<1>(range)},
          m_bin_edges(edges) {}

    /// @returns the total number of bins
    DETRAY_HOST_DEVICE
    dindex nbins() const { return m_n_bins; }

    /// Access function to a single bin from a value v
    ///
    /// @param v is the value for the bin search
    ///
    /// @returns the corresponding bin index
    DETRAY_HOST_DEVICE
    int bin(const scalar_type v) const {
        auto bins_begin =
            m_bin_edges->begin() + static_cast<index_type>(m_offset);
        auto bins_end = bins_begin + static_cast<index_type>(m_n_bins);

        return static_cast<int>(
                   detray::detail::lower_bound(bins_begin, bins_end, v) -
                   bins_begin) -
               1;
    }

    /// Access function to a range with binned neighborhood
    ///
    /// @note This is an inclusive range
    ///
    /// @param v is the value for the bin search
    /// @param nhood is the neighborhood range (# neighboring bins)
    ///
    /// @returns the corresponding range of bin indices
    DETRAY_HOST_DEVICE
    bin_range range(const scalar_type v,
                    const array_type<dindex, 2> &nhood) const {
        const int ibin{bin(v)};
        const int ibinmin{ibin - static_cast<int>(nhood[0])};
        const int ibinmax{ibin + static_cast<int>(nhood[1])};

        return {ibinmin, ibinmax};
    }

    /// Access function to a range with scalar neighborhood
    ///
    /// @param v is the value for the bin search
    /// @param nhood is the neighborhood range (range on axis values)
    ///
    /// @returns the corresponding range of bin indices
    DETRAY_HOST_DEVICE
    bin_range range(const scalar_type v,
                    const array_type<scalar_type, 2> &nhood) const {
        return {bin(v - nhood[0]), bin(v + nhood[1])};
    }

    /// @return the bin edges for a given @param ibin
    DETRAY_HOST_DEVICE
    array_type<scalar_type, 2> bin_edges(const dindex ibin) const {
        return {(*m_bin_edges)[m_offset + ibin],
                (*m_bin_edges)[m_offset + ibin + 1u]};
    }

    /// @return the values of the edges of all bins - uses dynamic memory
    // TODO: return range view instead to make it work properly in device
    DETRAY_HOST_DEVICE
    vector_type<scalar_type> bin_edges() const {
        // Transcribe the subvector for this binning from the global storage
        vector_type<scalar_type> edges;
        detray::detail::call_reserve(edges, nbins());

        edges.insert(edges.end(),
                     m_bin_edges->begin() + static_cast<index_type>(m_offset),
                     m_bin_edges->begin() +
                         static_cast<index_type>(m_offset + m_n_bins));

        return edges;
    }

    /// @return the bin width of a bin with index @param ibin.
    DETRAY_HOST_DEVICE
    scalar_type bin_width(const dindex ibin) const {

        // Get the binning information
        const array_type<scalar_type, 2> edges = bin_edges(ibin);

        return edges[1] - edges[0];
    }

    /// @return the span of the binning (equivalent to the span of the axis:
    /// [min, max) )
    DETRAY_HOST_DEVICE
    array_type<scalar_type, 2> span() const {
        // Get the binning information
        const scalar_type min{(*m_bin_edges)[m_offset]};
        const scalar_type max{(*m_bin_edges)[m_offset + m_n_bins]};

        return {min, max};
    }

    /// Equality operator
    ///
    /// @param rhs the axis to compare to
    ///
    /// @note as we cannot guarantee to have the same pointer for the bin edges,
    /// we make a fast comparison of the pointer first, but also allow for a
    /// value based comparison
    ///
    /// @returns whether the two axes are equal
    DETRAY_HOST_DEVICE constexpr bool operator==(const irregular &rhs) const {
        if (m_n_bins != rhs.m_n_bins) {
            return false;
        }
        if (m_offset == rhs.m_offset && m_bin_edges == rhs.m_bin_edges) {
            return true;
        }
        auto edge_range_lhs = detray::ranges::subrange(
            *m_bin_edges, dindex_range{m_offset, m_offset + m_n_bins});
        auto edge_range_rhs = detray::ranges::subrange(
            *rhs.m_bin_edges,
            dindex_range{rhs.m_offset, rhs.m_offset + rhs.m_n_bins});

        for (dindex i = 0; i < m_n_bins; ++i) {
            if (edge_range_lhs[i] != edge_range_rhs[i]) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace detray::axis
