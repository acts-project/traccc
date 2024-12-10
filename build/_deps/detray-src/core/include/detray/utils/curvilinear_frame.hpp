/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/geometry/coordinates/cartesian2D.hpp"
#include "detray/geometry/mask.hpp"
#include "detray/geometry/shapes/rectangle2D.hpp"
#include "detray/propagator/detail/jacobian_engine.hpp"
#include "detray/tracks/bound_track_parameters.hpp"
#include "detray/utils/unit_vectors.hpp"

namespace detray {

template <typename algebra_t>
struct curvilinear_frame {

    using transform3_type = dtransform3D<algebra_t>;
    using vector3 = typename transform3_type::vector3;
    using unit_vectors_type = unit_vectors<vector3>;
    using bound_to_free_matrix_type = bound_to_free_matrix<algebra_t>;
    using bound_vector_type = bound_parameters_vector<algebra_t>;
    using jacobian_engine_type =
        detail::jacobian_engine<cartesian2D<algebra_t>>;
    using free_track_parameters_type = free_track_parameters<algebra_t>;

    DETRAY_HOST_DEVICE
    explicit curvilinear_frame(const free_track_parameters_type& free_params) {

        const auto t = free_params.pos();
        const auto z = free_params.dir();
        const auto x = unit_vectors_type().make_curvilinear_unit_u(z);

        m_trf = transform3_type(t, z, x);
        m_bound_vec = detail::free_to_bound_vector<cartesian2D<algebra_t>>(
            m_trf, free_params);
    }

    DETRAY_HOST_DEVICE
    bound_to_free_matrix_type bound_to_free_jacobian() const {
        return jacobian_engine_type().bound_to_free_jacobian(
            m_trf, mask<rectangle2D>{}, m_bound_vec);
    }

    transform3_type m_trf{};
    bound_vector_type m_bound_vec{};

};  // curvilinear frame

}  // namespace detray
