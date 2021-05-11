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

template < typename T, class... Axes >
class grid{
public:
    /// number of dimensions of the grid
    static constexpr size_t DIM = sizeof...(Axes);

    /// type of values stored
    using value_type = T;
    /// reference type to values stored
    using reference = vecmem::vector<value_type>&;
    /// constant reference type to values stored
    using const_reference = const vecmem::vector<value_type>&;
    /// index type using local bin indices along each axis
    using index_t = std::array<size_t, DIM>;

    
    /// @brief default constructor
    ///
    /// @param [in] axes actual axis objects spanning the grid
    grid(std::tuple<Axes...> axes, vecmem::memory_resource* resource= nullptr):
	m_axes(std::move(axes)),
    	m_values(resource) {
	m_values.resize(size());
    }
    
    reference at(size_t bin) { return m_values.at(bin); }
    const_reference at(size_t bin) const { return m_values.at(bin); }
    
    size_t size() const {
	index_t nBinsArray = numLocalBins();
	// add under-and overflow bins for each axis and multiply all bins
	return std::accumulate(
			       nBinsArray.begin(), nBinsArray.end(), 1,
			       [](const size_t& a, const size_t& b) { return a * (b + 2); });
    }

    global_neighborhood_indices<DIM> neighborhood_indices(
	       const index_t& localBins, size_t size = 1u) const {
	return grid_helper::get_neighborhood_indices(localBins, size, m_axes);
    }
    
    index_t numLocalBins() const { return grid_helper::getNBins(m_axes); }
    
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

private:
    std::tuple< Axes... > m_axes;
    vecmem::jagged_vector<T> m_values;
};

} //namespace traccc
