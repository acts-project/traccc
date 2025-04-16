/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/rk_stepper.hpp>
#include <traccc/fitting/kalman_filter/kalman_fitter.hpp>

namespace traccc {
template <typename scalar_t>
using const_bfield_backend_t =
    ::covfie::backend::constant<::covfie::vector::vector_d<scalar_t, 3>,
                                ::covfie::vector::vector_d<scalar_t, 3>>;

template <typename scalar_t>
::covfie::field<const_bfield_backend_t<scalar_t>> construct_const_bfield(
    scalar_t x, scalar_t y, scalar_t z) {
    return ::covfie::field<const_bfield_backend_t<scalar_t>>{
        ::covfie::make_parameter_pack(
            typename const_bfield_backend_t<scalar_t>::configuration_t{x, y,
                                                                       z})};
}

template <typename scalar_t>
::covfie::field<const_bfield_backend_t<scalar_t>> construct_const_bfield(
    const vector3& v) {
    return construct_const_bfield(v[0], v[1], v[2]);
}
}  // namespace traccc
