/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/container.hpp"

// System include(s).
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace traccc {

/// Functor comparing two containers, and printing the results to the output
///
/// This is meant to be used in the example applications for nicely comparing
/// the results made on the host and on a device. Though the code actually
/// allows comparisons between any two containers.
///
/// @tparam HEADER_TYPE The header type in the container
/// @tparam ITEM_TYPE The item type in the container
///
template <typename HEADER_TYPE, typename ITEM_TYPE>
class container_comparator {

    public:
    /// Constructor with names for the output, and an output stream
    container_comparator(std::string_view type_name, std::string_view lhs_type,
                         std::string_view rhs_type,
                         std::ostream& out = std::cout,
                         const std::vector<scalar>& uncertainties = {
                             0.0001, 0.001, 0.01, 0.05});

    /// Function comparing two collections, and printing the results
    void operator()(
        const typename container_types<HEADER_TYPE, ITEM_TYPE>::const_view& lhs,
        const typename container_types<HEADER_TYPE, ITEM_TYPE>::const_view& rhs)
        const;

    private:
    /// Container type name to print
    std::string m_type_name;
    /// Type of the "Left Hand Side" collection
    std::string m_lhs_type;
    /// Type of the "Right Hand Side" collection
    std::string m_rhs_type;

    /// Output stream to print the results to
    std::reference_wrapper<std::ostream> m_out;

    /// Uncertainties to evaluate the comparison for
    std::vector<scalar> m_uncertainties;

};  // class container_comparator

}  // namespace traccc

// Include the implementation.
#include "traccc/performance/impl/container_comparator.ipp"
