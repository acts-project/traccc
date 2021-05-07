/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "utils/axis.hpp"

template < class... Axes >
class grid{
public:
    /// number of dimensions of the grid
    static constexpr size_t DIM = sizeof...(Axes);

    std::tuple< Axes... > m_axes;
    
    /// @brief default constructor
    ///
    /// @param [in] axes actual axis objects spanning the grid
    grid(std::tuple<Axes...> axes) : m_axes(std::move(axes)) {}

};
