/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// TODO: Remove this when gcc fixes their false positives.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic warning "-Wmaybe-uninitialized"
#endif

// Project include(s).
#include "detray/definitions/detail/math.hpp"
#include "detray/tracks/free_track_parameters.hpp"
#include "detray/utils/invalid_values.hpp"
#include "detray/utils/matrix_helper.hpp"

// System include(s).
#include <ostream>

namespace detray::detail {

/// @brief describes a helical trajectory in a given B-field.
///
/// Helix class for the analytical solution of track propagation in
/// homogeneous B field. This Follows the notation of Eq (4.7) in
/// DOI:10.1007/978-3-030-65771-0
template <typename algebra_t>
class helix {
    public:
    using algebra_type = algebra_t;
    using scalar_type = dscalar<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using transform3_type = dtransform3D<algebra_t>;
    using matrix_operator = dmatrix_operator<algebra_t>;

    /// Free track parameters
    using free_track_parameters_type = free_track_parameters<algebra_t>;

    /// 2D Matrix type
    template <std::size_t ROWS, std::size_t COLS>
    using matrix_type = dmatrix<algebra_t, ROWS, COLS>;
    using free_matrix_t = free_matrix<algebra_t>;
    using mat_helper = matrix_helper<matrix_operator>;

    DETRAY_HOST_DEVICE
    helix() = delete;

    /// Parametrized constructor
    ///
    /// @param pos the the origin of the helix
    /// @param time the time parameter
    /// @param dir the initial normalized direction for the helix
    /// @param q the charge of the particle
    /// @param mag_field the magnetic field vector
    DETRAY_HOST_DEVICE
    helix(const point3_type &pos, const scalar_type time,
          const vector3_type &dir, const scalar_type qop,
          vector3_type const *const mag_field)
        : _pos(pos), _time(time), _qop(qop), _mag_field(mag_field), _t0{dir} {

        // Normalized B field
        _h0 = vector::normalize(*_mag_field);

        assert((math::fabs(getter::norm(_t0) - 1.f) < 1e-5f) &&
               "The helix direction must be normalized");

        // Momentum
        const vector3_type mom =
            1.f / static_cast<scalar_type>(math::fabs(qop)) * _t0;

        // Normalized _h0 X _t0
        _n0 = vector::normalize(vector::cross(_h0, _t0));

        // Magnitude of _h0 X _t0
        _alpha = getter::norm(vector::cross(_h0, _t0));

        // Dot product of _h0 X _t0
        _delta = vector::dot(_h0, _t0);

        // B field strength
        _B = getter::norm(*_mag_field);

        // Path length scaler
        _K = -_qop * _B;

        // Get longitudinal momentum parallel to B field
        scalar_type pz = vector::dot(mom, _h0);

        // Get transverse momentum perpendicular to B field
        vector3_type pT = mom - pz * _h0;

        // R [mm] =  pT [GeV] / B [T] in natrual unit
        _R = getter::norm(pT) / _B;

        // Handle the case of pT ~ 0
        if (getter::norm(pT) < 1e-6f) {
            _vz_over_vt = detail::invalid_value<scalar_type>();
        } else {
            // Get vz over vt in new coordinate
            _vz_over_vt = pz / getter::norm(pT);
        }
    }

    DETRAY_HOST_DEVICE
    helix(const free_track_parameters_type &track,
          vector3_type const *const mag_field)
        : helix(track.pos(), track.time(), track.dir(), track.qop(),
                mag_field) {}

    /// @returns the radius of helix
    DETRAY_HOST_DEVICE
    scalar_type radius() const { return _R; }

    /// @returns the position after propagating the path length of s
    DETRAY_HOST_DEVICE
    point3_type operator()(const scalar_type s) const { return this->pos(s); }

    /// @returns the position after propagating the path length of s
    DETRAY_HOST_DEVICE
    point3_type pos(const scalar_type s) const {

        // Handle the case of pT ~ 0
        if (_vz_over_vt == detail::invalid_value<scalar_type>()) {
            return _pos + s * _h0;
        }

        point3_type ret = _pos;
        ret = ret + _delta / _K * (_K * s - math::sin(_K * s)) * _h0;
        ret = ret + math::sin(_K * s) / _K * _t0;
        ret = ret + _alpha / _K * (1.f - math::cos(_K * s)) * _n0;

        return ret;
    }

    DETRAY_HOST_DEVICE
    point3_type pos() const { return _pos; }

    /// @returns the tangential vector after propagating the path length of s
    DETRAY_HOST_DEVICE
    vector3_type dir(const scalar_type s) const {

        // Handle the case of pT ~ 0
        if (_vz_over_vt == detail::invalid_value<scalar_type>()) {
            return _t0;
        }

        vector3_type ret{0.f, 0.f, 0.f};

        ret = ret + _delta * (1 - math::cos(_K * s)) * _h0;
        ret = ret + math::cos(_K * s) * _t0;
        ret = ret + _alpha * math::sin(_K * s) * _n0;

        return vector::normalize(ret);
    }

    DETRAY_HOST_DEVICE
    point3_type dir() const { return _t0; }

    DETRAY_HOST_DEVICE
    scalar_type time() const { return _time; }

    DETRAY_HOST_DEVICE
    scalar_type qop() const { return _qop; }

    DETRAY_HOST_DEVICE
    auto b_field() const { return _mag_field; }

    /// @returns the transport jacobian after propagating the path length of s
    DETRAY_HOST_DEVICE
    free_matrix_t jacobian(const scalar_type s) const {

        free_matrix_t ret =
            matrix_operator().template zero<e_free_size, e_free_size>();

        const matrix_type<3, 3> I33 =
            matrix_operator().template identity<3, 3>();
        const matrix_type<3, 3> Z33 = matrix_operator().template zero<3, 3>();

        // Notations
        // r = position
        // t = direction
        // l = qoverp

        // Get drdr
        auto drdr = I33;
        matrix_operator().set_block(ret, drdr, e_free_pos0, e_free_pos0);

        // Get dtdr
        auto dtdr = Z33;
        matrix_operator().set_block(ret, dtdr, e_free_dir0, e_free_pos0);

        // Get drdt
        auto drdt = Z33;

        const scalar sin_ks = math::sin(_K * s);
        const scalar cos_ks = math::cos(_K * s);
        drdt = drdt + sin_ks / _K * I33;

        matrix_type<3, 1> H0 = matrix_operator().template zero<3, 1>();
        getter::element(H0, 0u, 0u) = _h0[0u];
        getter::element(H0, 1u, 0u) = _h0[1u];
        getter::element(H0, 2u, 0u) = _h0[2u];
        const matrix_type<1, 3> H0_T = matrix_operator().transpose(H0);
        const matrix_type<3, 3> H0H0_T = H0 * H0_T;

        drdt = drdt + (_K * s - sin_ks) / _K * H0H0_T;

        drdt = drdt +
               (cos_ks - 1.f) / _K * mat_helper().column_wise_cross(I33, _h0);

        matrix_operator().set_block(ret, drdt, e_free_pos0, e_free_dir0);

        // Get dtdt
        auto dtdt = Z33;
        dtdt = dtdt + cos_ks * I33;
        dtdt = dtdt + (1 - cos_ks) * H0H0_T;
        dtdt = dtdt - sin_ks * mat_helper().column_wise_cross(I33, _h0);

        matrix_operator().set_block(ret, dtdt, e_free_dir0, e_free_dir0);

        // Get drdl
        vector3_type drdl =
            1.f / _qop * (s * this->dir(s) + _pos - this->pos(s));

        matrix_operator().set_block(ret, drdl, e_free_pos0, e_free_qoverp);

        // Get dtdl
        vector3_type dtdl =
            -_B * s *
            (sin_ks * (H0H0_T - I33) * _t0 + cos_ks * vector::cross(_h0, _t0));

        matrix_operator().set_block(ret, dtdl, e_free_dir0, e_free_qoverp);

        // 3x3 and 7x7 element is 1 (Maybe?)
        matrix_operator().element(ret, e_free_time, e_free_time) = 1.f;
        matrix_operator().element(ret, e_free_qoverp, e_free_qoverp) = 1.f;

        return ret;
    }

    /// Print
    DETRAY_HOST
    friend std::ostream &operator<<(std::ostream &os, const helix &h) {
        os << "helix: ";
        os << "ori = [" << h._pos[0] << ", " << h._pos[1] << ", " << h._pos[2]
           << "], ";
        os << "dir = [" << h._t0[0] << ", " << h._t0[1] << ", " << h._t0[2]
           << "]" << std::endl;

        return os;
    }

    /// origin
    point3_type _pos;

    /// time
    scalar_type _time;

    /// qop
    scalar_type _qop;

    /// B field
    vector3_type const *_mag_field;

    /// B field strength
    scalar _B;

    /// Normalized b field
    vector3_type _h0;

    /// Normalized tangent vector
    vector3_type _t0;

    /// Normalized _h0 X _t0
    vector3_type _n0;

    /// Magnitude of _h0 X _t0
    scalar_type _alpha;

    /// Dot product of _h0 X _t0
    scalar_type _delta;

    /// Path length scaler
    scalar_type _K;

    /// Radius [mm] of helix
    scalar_type _R;

    /// Velocity in new z axis divided by transverse velocity
    scalar_type _vz_over_vt;
};

}  // namespace detray::detail
