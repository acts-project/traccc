/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray include(s).
#include "detray/plugins/algebra/eigen_definitions.hpp"

// VecMem include(s).
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <array>
#include <map>
#include <tuple>

#define __plugin algebra::eigen

namespace traccc {

    using scalar = TRACCC_CUSTOM_SCALARTYPE;

    template <typename value_type, unsigned int kDIM>
    using darray = std::array<value_type, kDIM>;

    template <typename value_type>
    using dvector = vecmem::vector<value_type>;

    template <typename value_type>
    using djagged_vector = vecmem::jagged_vector<value_type>;

    template <typename key_type, typename value_type>
    using dmap = std::map<key_type, value_type>;

    template< class... types>
    using dtuple = std::tuple<types ...>;

    namespace getter = algebra::getter;
    namespace vector = algebra::vector;

} //namespace traccc
