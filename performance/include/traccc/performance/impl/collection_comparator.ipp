/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/performance/details/is_same_object.hpp"

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/container.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <string_view>
#include <vector>

namespace traccc {

template <typename TYPE>
collection_comparator<TYPE>::collection_comparator(
    std::string_view type_name, details::comparator_factory<TYPE> comp_factory,
    std::string_view lhs_type, std::string_view rhs_type, std::ostream& out,
    const std::vector<scalar>& uncertainties)
    : m_type_name(type_name),
      m_lhs_type(lhs_type),
      m_rhs_type(rhs_type),
      m_comp_factory(comp_factory),
      m_out(out),
      m_uncertainties(uncertainties) {}

template <typename TYPE>
void collection_comparator<TYPE>::operator()(
    const typename collection_types<TYPE>::const_view& lhs,
    const typename collection_types<TYPE>::const_view& rhs) const {

    // Create device collections on top of the views.
    const typename collection_types<TYPE>::const_device lhs_coll{lhs},
        rhs_coll{rhs};

    // Print some basic output.
    m_out.get() << "Number of " << m_type_name << ": " << lhs_coll.size()
                << " (" << m_lhs_type << "), " << rhs_coll.size() << " ("
                << m_rhs_type << ")\n";

    // Calculate the agreements at various uncertainties.
    std::vector<scalar> agreements;
    agreements.reserve(m_uncertainties.size());
    for (scalar uncertainty : m_uncertainties) {
        // The number of matched items between the containers.
        std::size_t matched = 0;
        // Iterate over all elements of the LHS collection.
        for (const TYPE& obj : lhs_coll) {
            // Check if there's an equivalent element in the RHS collection.
            if (std::find_if(rhs_coll.begin(), rhs_coll.end(),
                             m_comp_factory.make_comparator(
                                 obj, uncertainty)) != rhs_coll.end()) {
                ++matched;
            }
        }
        // Calculate the agreement value.
        agreements.push_back(
            static_cast<scalar>(matched) /
            static_cast<scalar>(std::max(lhs_coll.size(), rhs_coll.size())) *
            100.);
    }
    assert(agreements.size() == m_uncertainties.size());

    // Now print them.
    m_out.get() << "  Matching rate(s):\n";
    for (std::size_t i = 0; i < m_uncertainties.size(); ++i) {
        m_out.get() << "    - " << agreements.at(i) << "% at "
                    << m_uncertainties.at(i) * 100. << "% uncertainty\n";
    }
    m_out.get() << std::flush;
}

}  // namespace traccc
