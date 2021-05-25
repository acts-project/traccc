/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "utils/axis.hpp"
#include "utils/grid_helper.hpp"

namespace traccc{

/// @brief class for describing a regular multi-dimensional grid
///
/// @tparam T    type of values stored inside the bins of the grid
/// @tparam Axes parameter pack of axis types defining the grid
///
/// Class describing a multi-dimensional, regular grid which can store objects
/// in its multi-dimensional bins. Bins are hyper-boxes and can be accessed
/// either by global bin index, local bin indices or position.
///
/// @note @c T must be default-constructible.    
template < class... Axes >
class grid{
public:
    /// number of dimensions of the grid
    static constexpr size_t DIM = sizeof...(Axes);

    /// index type using local bin indices along each axis
    using index_t = std::array<size_t, DIM>;
    
    /// @brief default constructor
    ///
    /// @param [in] axes actual axis objects spanning the grid
    grid(std::tuple<Axes...> axes):
	m_axes(std::move(axes)){}

    /// @brief total number of bins
    ///
    /// @return total number of bins in the grid
    ///
    /// @note This number contains under-and overflow bins along all axes.    
    size_t size(bool full_counter= true) const {
	index_t nBinsArray = numLocalBins();
	// add under-and overflow bins for each axis and multiply all bins
	return std::accumulate(
			       nBinsArray.begin(), nBinsArray.end(), 1,
			       [](const size_t& a, const size_t& b) { return a * (b + 2); });
    }
    
    /// @brief get global bin indices for neighborhood
    ///
    /// @param [in] localBins center bin defined by local bin indices along each
    ///                       axis
    /// @param [in] size      size of neighborhood determining how many adjacent
    ///                       bins along each axis are considered
    /// @return set of global bin indices for all bins in neighborhood
    ///
    /// @note Over-/underflow bins are included in the neighborhood.
    /// @note The @c size parameter sets the range by how many units each local
    ///       bin index is allowed to be varied. All local bin indices are
    ///       varied independently, that is diagonal neighbors are included.
    ///       Ignoring the truncation of the neighborhood size reaching beyond
    ///       over-/underflow bins, the neighborhood is of size \f$2 \times
    ///       \text{size}+1\f$ along each dimension.
    global_neighborhood_indices<DIM> neighborhood_indices(
	       const index_t& localBins, size_t size = 1u) const {
	return grid_helper::get_neighborhood_indices(localBins, size, m_axes);
    }

    /// @brief get number of bins along each specific axis
    ///
    /// @return array giving the number of bins along all axes
    ///
    /// @note Not including under- and overflow bins
    index_t numLocalBins() const { return grid_helper::getNBins(m_axes); }
    
    /// @brief determine global index for bin containing the given point
    ///
    /// @tparam Point any type with point semantics supporting component access
    ///               through @c operator[]
    ///
    /// @param  [in] point point to look up in the grid
    /// @return global index for bin containing the given point
    ///
    /// @pre The given @c Point type must represent a point in d (or higher)
    ///      dimensions where d is dimensionality of the grid.
    /// @note This could be a under-/overflow bin along one or more axes.    
    template <class Point>
    size_t globalBinFromPosition(const Point& point) const {
	return globalBinFromLocalBins(localBinsFromPosition(point));
    }

    /// @brief determine global bin index from local bin indices along each axis
    ///
    /// @param  [in] localBins local bin indices along each axis
    /// @return global index for bin defined by the local bin indices
    ///
    /// @pre All local bin indices must be a valid index for the corresponding
    ///      axis (including the under-/overflow bin for this axis).
    size_t globalBinFromLocalBins(const index_t& localBins) const {
	return grid_helper::getGlobalBin(localBins, m_axes);
    }

    /// @brief  determine local bin index for each axis from the given point
    ///
    /// @tparam Point any type with point semantics supporting component access
    ///               through @c operator[]
    ///
    /// @param  [in] point point to look up in the grid
    /// @return array with local bin indices along each axis (in same order as
    ///         given @c axes object)
    ///
    /// @pre The given @c Point type must represent a point in d (or higher)
    ///      dimensions where d is dimensionality of the grid.
    /// @note This could be a under-/overflow bin along one or more axes.    
    template <class Point>
    index_t localBinsFromPosition(const Point& point) const {
	return grid_helper::getLocalBinIndices(point, m_axes);
    }   

private:
    std::tuple< Axes... > m_axes;
};

} //namespace traccc
