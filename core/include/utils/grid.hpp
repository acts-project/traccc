/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "utils/axis.hpp"
#include "utils/grid_helper.hpp"


template < typename T, class... Axes >
class grid{
public:
    /// number of dimensions of the grid
    static constexpr size_t DIM = sizeof...(Axes);

    /// type of values stored
    using value_type = T;
    /// reference type to values stored
    using reference = value_type&;
    /// index type using local bin indices along each axis
    using index_t = std::array<size_t, DIM>;
    
    template <class Point>
    reference at_position(const Point& point) {
	return m_values.at(globalBinFromPosition(point));
    }

    template <class Point>
    size_t globalBinFromPosition(const Point& point) const {
	return globalBinFromLocalBins(localBinsFromPosition(point));
    }

    size_t globalBinFromLocalBins(const index_t& localBins) const {
	return grid_helper::getGlobalBin(localBins, m_axes);
  }
    
    template <class Point>
    index_t localBinsFromPosition(const Point& point) const {
	return grid_helper::getLocalBinIndices(point, m_axes);
    }
    
    /// @brief default constructor
    ///
    /// @param [in] axes actual axis objects spanning the grid
    grid(std::tuple<Axes...> axes) : m_axes(std::move(axes)) {}

private:
    std::tuple< Axes... > m_axes;
    std::vector<T> m_values;
};

