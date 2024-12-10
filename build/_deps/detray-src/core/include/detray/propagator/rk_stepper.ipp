/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "detray/geometry/tracking_volume.hpp"

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE inline void detray::rk_stepper<
    magnetic_field_t, algebra_t, constraint_t, policy_t, inspector_t>::state::
    advance_track(
        const detray::rk_stepper<magnetic_field_t, algebra_t, constraint_t,
                                 policy_t, inspector_t>::intermediate_state& sd,
        const material<scalar_type>* vol_mat_ptr) {

    const scalar_type h{this->step_size()};
    const scalar_type h_6{h * static_cast<scalar_type>(1. / 6.)};
    auto& track = (*this)();
    auto pos = track.pos();
    auto dir = track.dir();

    // Update the track parameters according to the equations of motion
    // Reference: Eq (82) of https://doi.org/10.1016/0029-554X(81)90063-X
    pos = pos + h * (sd.t[0u] + h_6 * (sd.dtds[0] + sd.dtds[1] + sd.dtds[2]));
    track.set_pos(pos);

    // Reference: Eq (82) of https://doi.org/10.1016/0029-554X(81)90063-X
    dir =
        dir + h_6 * (sd.dtds[0] + 2.f * (sd.dtds[1] + sd.dtds[2]) + sd.dtds[3]);
    dir = vector::normalize(dir);
    track.set_dir(dir);

    auto qop = track.qop();
    if (vol_mat_ptr != nullptr) {
        // Reference: Eq (82) of https://doi.org/10.1016/0029-554X(81)90063-X
        qop =
            qop + h_6 * (sd.dqopds[0u] + 2.f * (sd.dqopds[1u] + sd.dqopds[2u]) +
                         sd.dqopds[3u]);
    }
    track.set_qop(qop);

    // Update path lengths
    this->update_path_lengths(h);
}

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE inline void detray::rk_stepper<
    magnetic_field_t, algebra_t, constraint_t, policy_t,
    inspector_t>::state::advance_jacobian(const detray::stepping::config& cfg,
                                          const intermediate_state& sd,
                                          const material<scalar_type>*
                                              vol_mat_ptr) {
    /// The calculations are based on ATL-SOFT-PUB-2009-002. The update of the
    /// Jacobian matrix is requires only the calculation of eq. 17 and 18.
    /// Since the terms of eq. 18 are currently 0, this matrix is not needed
    /// in the calculation. The matrix A from eq. 17 consists out of 3
    /// different parts. The first one is given by the upper left 3x3 matrix
    /// that are calculated by the derivatives dF/dT (called dFdT) and dG/dT
    /// (calles dGdT). The second is given by the top 3 lines of the rightmost
    /// column. This is calculated by dFdqop and dGdqop. The remaining non-zero
    /// term is calculated directly. The naming of the variables is explained in
    /// eq. 11 and are directly related to the initial problem in eq. 7. The
    /// evaluation is based by propagating the parameters T and lambda as given
    /// in eq. 16 and evaluating the derivations for matrix A.
    /// @note The translation for u_{n+1} in eq. 7 is in this case a
    /// 3-dimensional vector without a dependency of Lambda or lambda neither in
    /// u_n nor in u_n'. The second and fourth eq. in eq. 14 have the constant
    /// offset matrices h * Id and Id respectively. This involves that the
    /// constant offset does not exist for rectangular matrix dGdu' (due to the
    /// missing Lambda part) and only exists for dFdu' in dlambda/dlambda.

    // Set transport matrix (D) and update Jacobian transport
    //( JacTransport = D * JacTransport )
    auto D = matrix_operator().template identity<e_free_size, e_free_size>();

    const scalar_type h{this->step_size()};
    auto& track = (*this)();

    // Half step length
    const scalar_type h2{h * h};
    const scalar_type half_h{h * 0.5f};
    const scalar_type h_6{h * (1.f / 6.f)};

    // 3X3 Identity matrix
    const matrix_type<3, 3> I33 = matrix_operator().template identity<3, 3>();

    // Initialize derivatives
    std::array<matrix_type<3u, 3u>, 4u> dkndt{I33, I33, I33, I33};
    std::array<vector3_type, 4u> dkndqop;
    std::array<matrix_type<3u, 3u>, 4u> dkndr;
    std::array<scalar_type, 4u> dqopn_dqop{1.f, 1.f, 1.f, 1.f};

    /*---------------------------------------------------------------------------
     *  dk_n/dt1
     *    = qop_n * (dt_n/dt1 X B_n)
     *      + qop_n * ( t_n X dB_n/dt1 ),
     *  where dB_n/dt1 == dB_n/dr_n * dr_n/dt1.
     *
     *  The second term is non-zero only for inhomogeneous magnetic fields
     *
     *  Note that [ t_n = t1 + h * d(t_{n-1})/ds) ] as indicated by
     *  Eq (84) of https://doi.org/10.1016/0029-554X(81)90063-X

     *  [ Table for dt_n/dt1 ]
     *  dt1/dt1 = I
     *  dt2/dt1 = d( t1 + h/2 * dt1/ds ) / dt1 = I + h/2 * dk1/dt1
     *  dt3/dt1 = d( t1 + h/2 * dt2/ds ) / dt1 = I + h/2 * dk2/dt1
     *  dt4/dt1 = d( t1 + h * dt3/ds ) / dt1 = I + h * dk3/dt1
     *
     *  [ Table for dr_n/dt1 ]
     *  dr1/dt1 = 0
     *  dr2/dt1 = d(r1 + h/2 * t1 + h^2/8 dt1/ds)/dt1 = h/2 * I + h^2/8 dk1/dt1
     *  dr3/dt1 = d(r1 + h/2 * t1 + h^2/8 dt1/ds)/dt1 = h/2 * I + h^2/8 dk1/dt1
     *  dr4/dt1 = d(r1 + h * t1 + h^2/2 dt3/ds)/dt1 = h * I + h^2/2 dk3/dt1
     *
     *  Note that
     *  d/dr [ F(T) X B ]  = dF(T)/dr (X) B, where (X) means the column wise
     *  cross product
    ---------------------------------------------------------------------------*/

    /*---------------------------------------------------------------------------
     *  dk_n/dqop_1
     *    = dqop_n/dqop1 * ( t_n X B_n )
     *      + qop_n * ( dt_n/dqop1 X B_n )
     *      + qop_n * ( t_n X dB_n/dqop1 ),
     *  where dB_n/dqop1 = dB_n/dr_n * dr_n/dqop1
     *
     *  Note that [ qop_n = qop1 + h * dqop_{n-1}/ds) ] as indicated by
     *  Eq (84) of https://doi.org/10.1016/0029-554X(81)90063-X
     *
     *  [ Table for dqop_n/dqop1 ]
     *  dqop1/dqop1 = 1
     *  dqop2/dqop1 = 1 + h/2 * d(dqop1/ds)/dqop1
     *  dqop3/dqop1 = 1 + h/2 * d(dqop2/ds)/dqop1
     *  dqop4/dqop1 = 1 + h * d(dqop3/ds)/dqop1
     *
     *  [ Table for dt_n/dqop1 ]
     *  dt1/dqop1 = 0
     *  dt2/dqop1 = d(t1 + h/2 dt1/ds)/dqop1 = h/2 * dk1/dqop1
     *  dt3/dqop1 = d(t1 + h/2 dt2/ds)/dqop1 = h/2 * dk2/dqop1
     *  dt4/dqop1 = d(t1 + h dt3/ds)/dqop1 = h * dk3/dqop1
     *
     *  [ Table for dr_n/dqop1 ]
     *  dr1/dqop1 = 0
     *  dr2/dqop1 = d(r1 + h/2 * t1 + h^2/8 dt1/ds)/dqop1 = h^2/8 * dk1/dqop1
     *  dr3/dqop1 = d(r1 + h/2 * t1 + h^2/8 dt1/ds)/dqop1 = h^2/8 * dk1/dqop1
     *  dr4/dqop1 = d(r1 + h * t1 + h^2/2 dt3/ds)/dqop1 = h^2/2 dk3/dqop1
    ---------------------------------------------------------------------------*/

    /*---------------------------------------------------------------------------
     *  dk_n/dr1
     *    = qop_n * ( dt_n/dr1 X B_n )
     *      + qop_n * ( t_n X dB_n/dr1 ),
     *  where dB_n/dr1 = dB_n/dr_n * dr_n/dr1
     *
     *  [ Table for dt_n/dr1 ]
     *  dt1/dr1 = 0
     *  dt2/dr1 = d(t1 + h/2 * dt1/ds)/dr1 = h/2 * dk1/dr1
     *  dt2/dr1 = d(t1 + h/2 * dt2/ds)/dr1 = h/2 * dk2/dr1
     *  dt3/dr1 = d(t1 + h * dt3/ds)/dr1 = h * dk3/dr1
     *
     *  [ Table for dr_n/dr1 ]
     *  dr1/dr1 = I
     *  dr2/dr1 = (r1 + h/2 * t1 + h^2/8 dt1/ds ) / dr1 = I + h^2/8 dk1/dr1
     *  dr3/dr1 = (r1 + h/2 * t1 + h^2/8 dt1/ds ) / dr1 = I + h^2/8 dk1/dr1
     *  dr4/dr1 = (r1 + h * t1 + h^2/2 dt3/ds ) / dr1 = I + h^2/2 dk3/dr1
    ---------------------------------------------------------------------------*/

    /*---------------------------------------------------------------------------
     *  d(dqop_n/ds)/dqop1
     *
     *  Useful equation:
     *  dqop/ds = qop^3 * E * (-dE/ds) / q^2 = - qop^3 * E g / q^2

     *  [ Table for d(dqop_n/ds)/dqop1 ]
     *  d(dqop1/ds)/dqop1 = dqop1/ds * (1/qop * (3 - p^2/E^2) + 1/g1 * dg1dqop1
     *  d(dqop2/ds)/dqop1 = d(dqop2/ds)/dqop2 * (1 + h/2 * d(dqop1/ds)/dqop1)
     *  d(dqop3/ds)/dqop1 = d(dqop3/ds)/dqop3 * (1 + h/2 * d(dqop2/ds)/dqop1)
     *  d(dqop4/ds)/dqop1 = d(dqop4/ds)/dqop4 * (1 + h * d(dqop3/ds)/dqop1)
    ---------------------------------------------------------------------------*/

    if (!cfg.use_eloss_gradient) {
        getter::element(D, e_free_qoverp, e_free_qoverp) = 1.f;
    } else {
        // Pre-calculate dqop_n/dqop1
        const scalar_type d2qop1dsdqop1 =
            this->d2qopdsdqop(sd.qop[0u], vol_mat_ptr);

        dqopn_dqop[0u] = 1.f;
        dqopn_dqop[1u] = 1.f + half_h * d2qop1dsdqop1;

        const scalar_type d2qop2dsdqop1 =
            this->d2qopdsdqop(sd.qop[1u], vol_mat_ptr) * dqopn_dqop[1u];
        dqopn_dqop[2u] = 1.f + half_h * d2qop2dsdqop1;

        const scalar_type d2qop3dsdqop1 =
            this->d2qopdsdqop(sd.qop[2u], vol_mat_ptr) * dqopn_dqop[2u];
        dqopn_dqop[3u] = 1.f + h * d2qop3dsdqop1;

        const scalar_type d2qop4dsdqop1 =
            this->d2qopdsdqop(sd.qop[3u], vol_mat_ptr) * dqopn_dqop[3u];

        /*-----------------------------------------------------------------
         * Calculate the first terms of d(dqop_n/ds)/dqop1
        -------------------------------------------------------------------*/

        getter::element(D, e_free_qoverp, e_free_qoverp) =
            1.f + h_6 * (d2qop1dsdqop1 + 2.f * (d2qop2dsdqop1 + d2qop3dsdqop1) +
                         d2qop4dsdqop1);
    }

    /*-----------------------------------------------------------------
     * Calculate the first terms of dk_n/dt1
    -------------------------------------------------------------------*/
    using mat_helper = matrix_helper<matrix_operator>;
    // dk1/dt1
    dkndt[0u] =
        sd.qop[0u] * mat_helper().column_wise_cross(dkndt[0u], sd.b_first);

    // dk2/dt1
    dkndt[1u] = dkndt[1u] + half_h * dkndt[0u];
    dkndt[1u] =
        sd.qop[1u] * mat_helper().column_wise_cross(dkndt[1u], sd.b_middle);

    // dk3/dt1
    dkndt[2u] = dkndt[2u] + half_h * dkndt[1u];
    dkndt[2u] =
        sd.qop[2u] * mat_helper().column_wise_cross(dkndt[2u], sd.b_middle);

    // dk4/dt1
    dkndt[3u] = dkndt[3u] + h * dkndt[2u];
    dkndt[3u] =
        sd.qop[3u] * mat_helper().column_wise_cross(dkndt[3u], sd.b_last);

    /*-----------------------------------------------------------------
     * Calculate the first and second terms of dk_n/dqop1
    -------------------------------------------------------------------*/
    // dk1/dqop1
    dkndqop[0u] = dqopn_dqop[0u] * vector::cross(sd.t[0u], sd.b_first);

    // dk2/dqop1
    dkndqop[1u] = dqopn_dqop[1u] * vector::cross(sd.t[1u], sd.b_middle) +
                  sd.qop[1u] * half_h * vector::cross(dkndqop[0u], sd.b_middle);

    // dk3/dqop1
    dkndqop[2u] = dqopn_dqop[2u] * vector::cross(sd.t[2u], sd.b_middle) +
                  sd.qop[2u] * half_h * vector::cross(dkndqop[1u], sd.b_middle);

    // dk4/dqop1
    dkndqop[3u] = dqopn_dqop[3u] * vector::cross(sd.t[3u], sd.b_last) +
                  sd.qop[3u] * h * vector::cross(dkndqop[2u], sd.b_last);

    // Calculate dkndr in case of considering B field gradient
    if (cfg.use_field_gradient) {

        // Positions and field gradients at initial, middle and final points of
        // the fourth order RKN
        vector3_type r_ini = track.pos();
        vector3_type r_mid =
            r_ini + half_h * sd.t[0u] + h2 * 0.125f * sd.dtds[0u];
        vector3_type r_fin = r_ini + h * sd.t[0u] + h2 * 0.5f * sd.dtds[2u];

        matrix_type<3, 3> dBdr_ini = evaluate_field_gradient(r_ini);
        matrix_type<3, 3> dBdr_mid = evaluate_field_gradient(r_mid);
        matrix_type<3, 3> dBdr_fin = evaluate_field_gradient(r_fin);

        /*-----------------------------------------------------------------
         * Calculate all terms of dk_n/dr1
        -------------------------------------------------------------------*/

        // dk1/dr1
        dkndr[0u] =
            -sd.qop[0u] * mat_helper().column_wise_cross(dBdr_ini, sd.t[0u]);

        // dk2/dr1
        dkndr[1u] = sd.qop[1u] * mat_helper().column_wise_cross(
                                     half_h * dkndr[0u], sd.b_middle);
        dkndr[1u] = dkndr[1u] -
                    sd.qop[1u] * mat_helper().column_wise_cross(
                                     dBdr_mid * (I33 + h2 * 0.125f * dkndr[0u]),
                                     sd.t[1u]);

        // dk3/dr1
        dkndr[2u] = sd.qop[2u] * mat_helper().column_wise_cross(
                                     half_h * dkndr[1u], sd.b_middle);
        dkndr[2u] = dkndr[2u] -
                    sd.qop[2u] * mat_helper().column_wise_cross(
                                     dBdr_mid * (I33 + h2 * 0.125f * dkndr[0u]),
                                     sd.t[2u]);

        // dk4/dr1
        dkndr[3u] = sd.qop[3u] *
                    mat_helper().column_wise_cross(h * dkndr[2u], sd.b_last);
        dkndr[3u] = dkndr[3u] -
                    sd.qop[3u] *
                        mat_helper().column_wise_cross(
                            dBdr_fin * (I33 + h2 * 0.5f * dkndr[2u]), sd.t[3u]);

        // Set dF/dr1 and dG/dr1
        auto dFdr = matrix_operator().template identity<3, 3>();
        auto dGdr = matrix_operator().template identity<3, 3>();
        dFdr = dFdr + h * h_6 * (dkndr[0u] + dkndr[1u] + dkndr[2u]);
        dGdr = h_6 * (dkndr[0u] + 2.f * (dkndr[1u] + dkndr[2u]) + dkndr[3u]);

        matrix_operator().set_block(D, dFdr, 0u, 0u);
        matrix_operator().set_block(D, dGdr, 4u, 0u);
    }

    // Set dF/dt1 and dG/dt1
    auto dFdt = matrix_operator().template identity<3, 3>();
    auto dGdt = matrix_operator().template identity<3, 3>();
    dFdt = dFdt + h_6 * (dkndt[0u] + dkndt[1u] + dkndt[2u]);
    dFdt = h * dFdt;
    dGdt = dGdt + h_6 * (dkndt[0u] + 2.f * (dkndt[1u] + dkndt[2u]) + dkndt[3u]);

    matrix_operator().set_block(D, dFdt, 0u, 4u);
    matrix_operator().set_block(D, dGdt, 4u, 4u);

    // Set dF/dqop1 and dG/dqop1
    vector3_type dFdqop = h * h_6 * (dkndqop[0u] + dkndqop[1u] + dkndqop[2u]);
    vector3_type dGdqop =
        h_6 * (dkndqop[0u] + 2.f * (dkndqop[1u] + dkndqop[2u]) + dkndqop[3u]);
    matrix_operator().set_block(D, dFdqop, 0u, 7u);
    matrix_operator().set_block(D, dGdqop, 4u, 7u);

    this->set_transport_jacobian(D * this->transport_jacobian());
}

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE inline auto detray::rk_stepper<
    magnetic_field_t, algebra_t, constraint_t, policy_t, inspector_t>::state::
    evaluate_dqopds(const std::size_t i, const scalar_type h,
                    const scalar_type dqopds_prev,
                    const material<scalar_type>* vol_mat_ptr,
                    const detray::stepping::config& cfg)
        -> detray::pair<scalar_type, scalar_type> {

    const auto& track = (*this)();

    if (!vol_mat_ptr) {
        const scalar_type qop = track.qop();
        return detray::make_pair(scalar_type(0.f), qop);
    } else if (cfg.use_mean_loss && i != 0u) {
        // qop_n is calculated recursively like the direction of
        // evaluate_dtds.
        //
        // https://doi.org/10.1016/0029-554X(81)90063-X says:
        // "For y  we  have  similar  formulae  as  for x, for y' and
        // \lambda similar  formulae as for  x'"
        const scalar_type qop = track.qop() + h * dqopds_prev;
        return detray::make_pair(this->dqopds(qop, vol_mat_ptr), qop);
    } else {
        const scalar_type qop = track.qop();
        return detray::make_pair(this->dqopds(qop, vol_mat_ptr), qop);
    }
}

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE inline auto detray::rk_stepper<
    magnetic_field_t, algebra_t, constraint_t, policy_t,
    inspector_t>::state::evaluate_dtds(const vector3_type& b_field,
                                       const std::size_t i, const scalar_type h,
                                       const vector3_type& dtds_prev,
                                       const scalar_type qop)
    -> detray::pair<vector3_type, vector3_type> {
    auto& track = (*this)();
    const auto dir = track.dir();

    // Eq (84) of https://doi.org/10.1016/0029-554X(81)90063-X
    vector3_type t{(i == 0u) ? dir : dir + h * dtds_prev};

    // dtds = qop * (t X B) from Lorentz force
    return detray::make_pair(vector3_type{qop * vector::cross(t, b_field)}, t);
}

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE inline auto detray::rk_stepper<
    magnetic_field_t, algebra_t, constraint_t, policy_t,
    inspector_t>::state::evaluate_field_gradient(const point3_type& pos)
    -> matrix_type<3, 3> {

    matrix_type<3, 3> dBdr = matrix_operator().template zero<3, 3>();

    constexpr auto delta{1e-1f * unit<scalar_type>::mm};

    for (unsigned int i = 0; i < 3; i++) {

        point3_type dpos1 = pos;
        dpos1[i] += delta;
        const auto bvec1_tmp =
            this->m_magnetic_field.at(dpos1[0], dpos1[1], dpos1[2]);
        vector3_type bvec1;
        bvec1[0u] = bvec1_tmp[0u];
        bvec1[1u] = bvec1_tmp[1u];
        bvec1[2u] = bvec1_tmp[2u];

        point3_type dpos2 = pos;
        dpos2[i] -= delta;
        const auto bvec2_tmp =
            this->m_magnetic_field.at(dpos2[0], dpos2[1], dpos2[2]);
        vector3_type bvec2;
        bvec2[0u] = bvec2_tmp[0u];
        bvec2[1u] = bvec2_tmp[1u];
        bvec2[2u] = bvec2_tmp[2u];

        const vector3_type gradient = (bvec1 - bvec2) * (1.f / (2.f * delta));

        getter::element(dBdr, 0u, i) = gradient[0u];
        getter::element(dBdr, 1u, i) = gradient[1u];
        getter::element(dBdr, 2u, i) = gradient[2u];
    }

    return dBdr;
}

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE inline auto
detray::rk_stepper<magnetic_field_t, algebra_t, constraint_t, policy_t,
                   inspector_t>::state::dtds() const -> vector3_type {

    // In case there was no step before
    if (this->path_length() == 0.f) {
        const point3_type pos = (*this)().pos();

        const auto bvec_tmp = this->m_magnetic_field.at(pos[0], pos[1], pos[2]);
        vector3_type bvec;
        bvec[0u] = bvec_tmp[0u];
        bvec[1u] = bvec_tmp[1u];
        bvec[2u] = bvec_tmp[2u];

        return (*this)().qop() * vector::cross((*this)().dir(), bvec);
    }

    return m_dtds_3;
}

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE inline auto detray::rk_stepper<
    magnetic_field_t, algebra_t, constraint_t, policy_t,
    inspector_t>::state::dqopds(const material<scalar_type>* vol_mat_ptr) const
    -> scalar_type {

    // In case there was no step before
    if (this->path_length() == 0.f) {
        return this->dqopds((*this)().qop(), vol_mat_ptr);
    }

    return m_dqopds_3;
}

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE auto detray::rk_stepper<
    magnetic_field_t, algebra_t, constraint_t, policy_t,
    inspector_t>::state::dqopds(const scalar_type qop,
                                const material<scalar_type>* vol_mat_ptr) const
    -> scalar_type {

    // d(qop)ds is zero for empty space
    if (!vol_mat_ptr) {
        return 0.f;
    }

    const scalar_type q = this->particle_hypothesis().charge();
    const scalar_type p = q / qop;
    const scalar_type mass = this->particle_hypothesis().mass();
    const scalar_type E = math::sqrt(p * p + mass * mass);

    // Compute stopping power
    const scalar_type stopping_power =
        interaction<scalar_type>().compute_stopping_power(
            *vol_mat_ptr, this->particle_hypothesis(), {mass, qop, q});

    // Assert that a momentum is a positive value
    assert(p >= 0.f);

    // d(qop)ds, which is equal to (qop) * E * (-dE/ds) / p^2
    // or equal to (qop)^3 * E * (-dE/ds) / q^2
    return qop * qop * qop * E * stopping_power / (q * q);
}

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE auto
detray::rk_stepper<magnetic_field_t, algebra_t, constraint_t, policy_t,
                   inspector_t>::state::d2qopdsdqop(const scalar_type qop,
                                                    const material<scalar_type>*
                                                        vol_mat_ptr) const
    -> scalar_type {

    if (!vol_mat_ptr) {
        return 0.f;
    }

    const scalar_type q = this->particle_hypothesis().charge();
    const scalar_type p = q / qop;
    const scalar_type p2 = p * p;

    const auto& mass = this->particle_hypothesis().mass();
    const scalar_type E2 = p2 + mass * mass;

    // Interaction object
    interaction<scalar_type> I;

    // g = dE/ds = -1 * (-dE/ds) = -1 * stopping power
    const detail::relativistic_quantities<scalar_type> rq(mass, qop, q);
    const scalar_type g =
        -1.f *
        I.compute_stopping_power(*vol_mat_ptr, this->particle_hypothesis(), rq);

    // dg/d(qop) = -1 * derivation of stopping power
    const scalar_type dgdqop =
        -1.f *
        I.derive_stopping_power(*vol_mat_ptr, this->particle_hypothesis(), rq);

    // d(qop)/ds = - qop^3 * E * g / q^2
    const scalar_type dqopds = this->dqopds(qop, vol_mat_ptr);

    // Check Eq 3.12 of
    // (https://iopscience.iop.org/article/10.1088/1748-0221/4/04/P04016/meta)
    return dqopds * (1.f / qop * (3.f - p2 / E2) + 1.f / g * dgdqop);
}

template <typename magnetic_field_t, typename algebra_t, typename constraint_t,
          typename policy_t, typename inspector_t>
DETRAY_HOST_DEVICE inline bool detray::rk_stepper<
    magnetic_field_t, algebra_t, constraint_t, policy_t, inspector_t>::
    step(const scalar_type dist_to_next,
         detray::rk_stepper<magnetic_field_t, algebra_t, constraint_t, policy_t,
                            inspector_t>::state& stepping,
         const detray::stepping::config& cfg, const bool do_reset,
         const material<scalar_type>* vol_mat_ptr) const {

    // Get stepper and navigator states
    auto& magnetic_field = stepping.m_magnetic_field;

    if (do_reset) {
        stepping.set_step_size(dist_to_next);
    } else if (stepping.step_size() > 0) {
        stepping.set_step_size(
            math::min(stepping.next_step_size(), dist_to_next));
    } else {
        stepping.set_step_size(
            math::max(stepping.next_step_size(), dist_to_next));
    }

    const point3_type pos = stepping().pos();

    intermediate_state sd{};

    // First Runge-Kutta point
    const auto bvec = magnetic_field.at(pos[0], pos[1], pos[2]);
    sd.b_first[0] = bvec[0];
    sd.b_first[1] = bvec[1];
    sd.b_first[2] = bvec[2];

    // qop should be recalcuated at every point
    // Reference: Eq (84) of https://doi.org/10.1016/0029-554X(81)90063-X
    detray::tie(sd.dqopds[0u], sd.qop[0u]) =
        stepping.evaluate_dqopds(0u, 0.f, 0.f, vol_mat_ptr, cfg);
    detray::tie(sd.dtds[0u], sd.t[0u]) = stepping.evaluate_dtds(
        sd.b_first, 0u, 0.f, vector3_type{0.f, 0.f, 0.f}, sd.qop[0u]);

    /// RKN step trial and error estimation
    const auto estimate_error = [&](const scalar_type& h) {
        // State the square and half of the step size
        const scalar_type h2{h * h};
        const scalar_type half_h{h * 0.5f};

        // Second Runge-Kutta point
        // qop should be recalcuated at every point
        // Eq (84) of https://doi.org/10.1016/0029-554X(81)90063-X
        const point3_type pos1 =
            pos + half_h * sd.t[0u] + h2 * 0.125f * sd.dtds[0u];
        const auto bvec1 = magnetic_field.at(pos1[0], pos1[1], pos1[2]);
        sd.b_middle[0] = bvec1[0];
        sd.b_middle[1] = bvec1[1];
        sd.b_middle[2] = bvec1[2];

        detray::tie(sd.dqopds[1u], sd.qop[1u]) = stepping.evaluate_dqopds(
            1u, half_h, sd.dqopds[0u], vol_mat_ptr, cfg);
        detray::tie(sd.dtds[1u], sd.t[1u]) = stepping.evaluate_dtds(
            sd.b_middle, 1u, half_h, sd.dtds[0u], sd.qop[1u]);

        // Third Runge-Kutta point
        // qop should be recalcuated at every point
        // Reference: Eq (84) of https://doi.org/10.1016/0029-554X(81)90063-X
        detray::tie(sd.dqopds[2u], sd.qop[2u]) = stepping.evaluate_dqopds(
            2u, half_h, sd.dqopds[1u], vol_mat_ptr, cfg);
        detray::tie(sd.dtds[2u], sd.t[2u]) = stepping.evaluate_dtds(
            sd.b_middle, 2u, half_h, sd.dtds[1u], sd.qop[2u]);

        // Last Runge-Kutta point
        // qop should be recalcuated at every point
        // Eq (84) of https://doi.org/10.1016/0029-554X(81)90063-X
        const point3_type pos2 = pos + h * sd.t[0u] + h2 * 0.5f * sd.dtds[2u];
        const auto bvec2 = magnetic_field.at(pos2[0], pos2[1], pos2[2]);
        sd.b_last[0] = bvec2[0];
        sd.b_last[1] = bvec2[1];
        sd.b_last[2] = bvec2[2];

        detray::tie(sd.dqopds[3u], sd.qop[3u]) =
            stepping.evaluate_dqopds(3u, h, sd.dqopds[2u], vol_mat_ptr, cfg);
        detray::tie(sd.dtds[3u], sd.t[3u]) =
            stepping.evaluate_dtds(sd.b_last, 3u, h, sd.dtds[2u], sd.qop[3u]);

        // Compute and check the local integration error estimate
        // @Todo
        constexpr const scalar_type one_sixth{
            static_cast<scalar_type>(1. / 6.)};
        const vector3_type err_vec =
            one_sixth * h2 *
            (sd.dtds[0u] - sd.dtds[1u] - sd.dtds[2u] + sd.dtds[3u]);

        return getter::norm(err_vec);
    };

    /// Calculate the scale factor for the stepsize adjustment using the error
    /// estimate @param err
    const auto step_size_scaling =
        [&cfg](const scalar_type& err) -> scalar_type {
        return static_cast<scalar_type>(
            math::min(math::max(math::sqrt(math::sqrt(cfg.rk_error_tol / err)),
                                static_cast<scalar_type>(0.25)),
                      static_cast<scalar_type>(4.)));
    };

    scalar_type error{1e20f};

    // If the estimated error is larger than the tolerance with an additional
    // margin, reduce the step size and try again
    const auto n_trials{cfg.max_rk_updates};
    for (unsigned int i = 0u; i < n_trials; i++) {
        stepping.count_trials();

        error = math::max(estimate_error(stepping.step_size()),
                          static_cast<scalar_type>(1e-20));

        // Error is small enough
        // ---> break and advance track
        if (error <= 4.f * cfg.rk_error_tol) {
            break;
        }
        // Error estimate is too big
        // ---> Make step size smaller and estimate error again
        else {
            stepping.set_step_size(stepping.step_size() *
                                   step_size_scaling(error));

            // Run inspection while the stepsize is getting adjusted
            stepping.run_inspector(cfg, "Adjust stepsize: ", i,
                                   step_size_scaling(error));
        }
    }

    stepping.m_dtds_3 = sd.dtds[3u];
    stepping.m_dqopds_3 = sd.dqopds[3u];

    // Check constraints
    if (const scalar_type max_step =
            stepping.constraints().template size<>(stepping.direction());
        math::fabs(stepping.step_size()) > math::fabs(max_step)) {

        // Run inspection before step size is cut
        stepping.run_inspector(cfg, "Before constraint: ");

        stepping.set_step_size(max_step);
    }

    // Advance track state
    stepping.advance_track(sd, vol_mat_ptr);

    // Advance jacobian transport
    if (cfg.do_covariance_transport) {
        stepping.advance_jacobian(cfg, sd, vol_mat_ptr);
    }

    // The step size estimation fot the next step
    stepping.set_next_step_size(stepping.step_size() *
                                step_size_scaling(error));

    // Don't allow a too small step size (unless requested by the navigator)
    if (stepping.next_step_size() < cfg.min_stepsize) {
        stepping.set_next_step_size(cfg.min_stepsize);
    }

    // Run final inspection
    stepping.run_inspector(cfg, "Step complete: ");

    return true;
}
