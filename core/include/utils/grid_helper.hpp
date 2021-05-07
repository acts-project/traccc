/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "utils/axis.hpp"

template <size_t N>
struct grid_helper_impl;

template <size_t N>
struct grid_helper_impl{

    template <class... Axes>
    static void getGlobalBin(const std::array<size_t, sizeof...(Axes)>& localBins,
			     const std::tuple<Axes...>& axes, size_t& bin,
			     size_t& area) {
	const auto& thisAxis = std::get<N>(axes);
	bin += area * localBins.at(N);
	// make sure to account for under-/overflow bins
	area *= (thisAxis.getNBins() + 2);
	grid_helper_impl<N - 1>::getGlobalBin(localBins, axes, bin, area);
    }    
    
    template <class Point, class... Axes>
    static void getLocalBinIndices(const Point& point,
				   const std::tuple<Axes...>& axes,
				   std::array<size_t, sizeof...(Axes)>& indices) {
	const auto& thisAxis = std::get<N>(axes);
	indices.at(N) = thisAxis.getBin(point[N]);
	grid_helper_impl<N - 1>::getLocalBinIndices(point, axes, indices);
    }
};
    
template <>
struct grid_helper_impl<0u> {

  template <class... Axes>
  static void getGlobalBin(const std::array<size_t, sizeof...(Axes)>& localBins,
                           const std::tuple<Axes...>& /*axes*/, size_t& bin,
                           size_t& area) {
    bin += area * localBins.at(0u);
  }

  template <class Point, class... Axes>
  static void getLocalBinIndices(const Point& point,
                                 const std::tuple<Axes...>& axes,
                                 std::array<size_t, sizeof...(Axes)>& indices) {
    const auto& thisAxis = std::get<0u>(axes);
    indices.at(0u) = thisAxis.getBin(point[0u]);
  }
    
};
    

struct grid_helper {
  template <class... Axes>
  static size_t getGlobalBin(
      const std::array<size_t, sizeof...(Axes)>& localBins,
      const std::tuple<Axes...>& axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;
    size_t area = 1;
    size_t bin = 0;

    grid_helper_impl<MAX>::getGlobalBin(localBins, axes, bin, area);

    return bin;
  }
    
  template <class Point, class... Axes>
  static std::array<size_t, sizeof...(Axes)> getLocalBinIndices(
      const Point& point, const std::tuple<Axes...>& axes) {
    constexpr size_t MAX = sizeof...(Axes) - 1;
    std::array<size_t, sizeof...(Axes)> indices;

    grid_helper_impl<MAX>::getLocalBinIndices(point, axes, indices);

    return indices;
  }    
};
