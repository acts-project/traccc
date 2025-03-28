/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_config.hpp"

// detray include(s).
#include "detray/tracks/bound_track_parameters.hpp"

// System include(s).
#include <limits>
#include <iostream>
#include <string>

namespace traccc {

    /// Triplet fitting algorithm to fit a single track

    template <typename detector_t, typename bfield_t>
    class triplet_fitter {

        public:
        // Algebra type
        using algebra_type = typename detector_t::algebra_type;

        // Vector type
        template <typename T>
        using vector_type = typename detector_t::template vector_type<T>;

        // Configuration type
        using config_type = fitting_config;

        // Matrix types
        using matrix_operator = detray::dmatrix_operator<algebra_type>;
        using size_type = detray::dsize_type<algebra_type>;
        template <size_type ROWS, size_type COLS>
        using matrix_type = detray::dmatrix<algebra_type, ROWS, COLS>;

        /// Constructor with a detector
        ///
        /// @param det the detector object
        /// @param field magnetic field
        /// @param cfg fitter configuration
        TRACCC_HOST_DEVICE
        triplet_fitter(const detector_t& det, const bfield_t& field, const config_type& cfg)
                    : m_detector(det), m_field(field), m_cfg(cfg) {}

        // Triplet struct
        struct triplet {

            /// Default construct
            triplet() = default;

            /// Construct with three hits
            ///
            /// @param hit_0 position of first hit
            /// @param hit_1 position of second hit
            /// @param hit_2 position of third hit
            triplet(const point3& hit_0, const point3& hit_1, const point3& hit_2)
                : m_hit_0(hit_0), m_hit_1(hit_1), m_hit_2(hit_2) { }

            // global positions of three hits
            point3 m_hit_0;
            point3 m_hit_1;
            point3 m_hit_2;

            // triplet parameters
            scalar m_phi_0{};
            scalar m_theta_0{};
            scalar m_rho_phi{};
            scalar m_rho_theta{};

            // hit position derivatives
            vecmem::vector<scalar> m_h_thet;
            vecmem::vector<scalar> m_h_phi;

            // Measurements for getting variances (hit shifts)
            // and surface orientation (scattering estimation)
            std::array<measurement, 3u> m_meas;
            
            // Estimated values
            scalar m_sigma_MS{}; // MS uncertainty
            scalar m_theta{}; // Polar angle 

        };

        /// Helper function - Initialize fitter
        ///
        /// @param in_track_states input track states (from measurements)
        TRACCC_HOST_DEVICE
        void init_fitter(const vector_type<track_state<algebra_type>>& in_track_states) {

            m_track_states = in_track_states;

            // Clear triplets from the last track
            m_triplets.clear();
        }

        /// Helper function - Make triplets 
        ///
        /// Makes triplets from consecutive measurements on the track
        ///
        TRACCC_HOST_DEVICE 
        void make_triplets() {

            // std::cout << "Making triplets\n";
            
            std::size_t n_triplets = m_track_states.size() - 2;

            m_triplets.reserve(n_triplets);

            // loop over measurements (track states) in candidate
            for (std::size_t i = 0; i < n_triplets; ++i) {

                // Get track states (and measurements)
                const track_state<algebra_type>& state_0 = m_track_states[i];
                const track_state<algebra_type>& state_1 = m_track_states[i+1];
                const track_state<algebra_type>& state_2 = m_track_states[i+2];
                
                const measurement& meas_0 = state_0.get_measurement();
                const measurement& meas_1 = state_1.get_measurement();
                const measurement& meas_2 = state_2.get_measurement();

                // Get surfaces
                detray::tracking_surface meas_0_sf(m_detector, meas_0.surface_link);
                detray::tracking_surface meas_1_sf(m_detector, meas_1.surface_link);
                detray::tracking_surface meas_2_sf(m_detector, meas_2.surface_link);

                point2 loc_2d_0{meas_0.local[0], meas_0.local[1]};
                point2 loc_2d_1{meas_1.local[0], meas_1.local[1]};
                point2 loc_2d_2{meas_2.local[0], meas_2.local[1]};

                // Convert to global
                point3 glob_3d_0 = meas_0_sf.bound_to_global({}, loc_2d_0, {});
                point3 glob_3d_1 = meas_1_sf.bound_to_global({}, loc_2d_1, {});
                point3 glob_3d_2 = meas_2_sf.bound_to_global({}, loc_2d_2, {});

                /*
                // Print global positions of measurements
                std::cout << glob_3d_0[0] << " " << glob_3d_0[1] << " " << glob_3d_0[2] << std::endl;
                if (i == n_triplets - 1) {
                    std::cout << glob_3d_1[0] << " " << glob_3d_1[1] << " " << glob_3d_1[2] << std::endl;
                    std::cout << glob_3d_2[0] << " " << glob_3d_2[1] << " " << glob_3d_2[2] << std::endl;
                }*/

                // Make triplet
                triplet t(glob_3d_0, glob_3d_1, glob_3d_2);
                
                // measurements copied here
                t.m_meas[0] = meas_0;
                t.m_meas[1] = meas_1;
                t.m_meas[2] = meas_2;
                
                // copy again
                m_triplets.push_back(t);

            }

            // std::cout << m_triplets.size() << " triplets made\n";

        }


        /// Helper function - Linearize triplet
        ///
        /// Calculates triplet parameters by linearizing around circle solution
        /// Estimates triplet polar angle and multiple scattering uncertainty
        ///
        /// @param t Triplet to linearize
        ///
        TRACCC_HOST_DEVICE void linearize_triplet(triplet& t) {

            // std::cout << "Linearization:\n";
            // Vectors joining hits
            vector3 x_01 {t.m_hit_1 - t.m_hit_0};
            vector3 x_12 {t.m_hit_2 - t.m_hit_1};
            vector3 x_02 {t.m_hit_2 - t.m_hit_0};

            // Transverse distances
            scalar d_01 = getter::perp(x_01);
            scalar d_12 = getter::perp(x_12);
            scalar d_02 = getter::perp(x_02);
            // std::cout << "d01 " << d_01 << " d12 " << d_12 << " d02 " << d_02 << std::endl;

            // Longitudinal distances
            scalar z_01 = x_01[2];
            scalar z_12 = x_12[2];
            // std::cout << "z01 " << z_01 << " z12 " << z_12 << std::endl;

            // Calculation of circle curvature and hence the entire 
            // linearization will fail for very low (or 0) transverse 
            // distances between hits. The default initialized (0)
            // values of triplet parameters are returned in this case.
            constexpr scalar d_transverse_lim = 10e-6f;

            // Curvature of circle in transverse plane
            scalar c_perp;
            if ((d_01 > d_transverse_lim and d_12 > d_transverse_lim and d_02 > d_transverse_lim)) {
                // TODO: x-prod evaluates to -ve, might have to be reversed
                c_perp = 2.f * math::fabs((vector::cross(x_01, x_12))[2]) / (d_01 * d_12 * d_02);
            }
            else {
                return;
            } 

            // std::cout << "\tc_perp " << c_perp << std::endl;


            // Parameters of the arc segments
            
            // Bending angles
            scalar phi_1C = 2.f * math::asin(0.5f * d_01 * c_perp);
            scalar phi_2C = 2.f * math::asin(0.5f * d_12 * c_perp);
            // std::cout << "phi1 " << phi_1C << " phi2 " << phi_2C << std::endl;

            // Polar angles
            scalar theta_1C = math::atan2(d_01 * 0.5f * phi_1C, z_01 * math::sin(0.5f * phi_1C));
            scalar theta_2C = math::atan2(d_12 * 0.5f * phi_2C, z_12 * math::sin(0.5f * phi_2C));
            
            // Adapt for c_perp = 0 case
            if (c_perp == 0.f) {
                theta_1C = math::atan2(d_01, z_01);
                theta_2C = math::atan2(d_12, z_12);
            }
            
            // Estimate polar angle of the triplet
            // from the polar angles of the two segments
            t.m_theta = 0.5f * (theta_1C + theta_2C);

            // Store frequently used trigonometric expressions
            // theta_1C
            scalar sin_theta1C = math::sin(theta_1C);
            scalar cos_theta1C = math::cos(theta_1C);
            scalar sin2_theta1C = sin_theta1C * sin_theta1C;
            // theta_2C
            scalar sin_theta2C = math::sin(theta_2C);
            scalar cos_theta2C = math::cos(theta_2C);
            scalar sin2_theta2C = sin_theta2C * sin_theta2C;

            
            // Find direction of track at scattering plane 
            // (using hits 0, 2 and the circle solution)
            vector3 tangent3D;

            // Mid-point
            vector2 m{0.5f * (t.m_hit_0[0] + t.m_hit_2[0]), 0.5f * (t.m_hit_0[1] + t.m_hit_2[1])};
            
            // Direction perpendicular to vector joining hits 0 & 2
            vector2 n{(t.m_hit_2[1] - t.m_hit_0[1]) / d_02, (t.m_hit_0[0] - t.m_hit_2[0]) / d_02};

            scalar perp_d = math::sqrt(1.f / (c_perp * c_perp) - 0.25f * (d_02 * d_02));

            // Two centres possible
            std::array<vector2, 2u> c;

            c[0] = vector2{m[0] + n[0] * perp_d, m[1] + n[1] * perp_d};
            c[1] = vector2{m[0] - n[0] * perp_d, m[1] - n[1] * perp_d};
            
            vector2 x1{t.m_hit_1[0], t.m_hit_1[1]};
            
            // Choose the correct centre
            vector2 c_correct{0.f, 0.f};
            for (const vector2& c_i : c) {
                // Centre of the circle cannot be
                // on the same side of the line 
                // connecting hits 0 & 2 as hit 1
                if (vector::dot(x1 - m, c_i - m) < 0.f) {
                    c_correct = c_i;
                    break;
                }
            }

            // std::cout << "c_correct " << c_correct[0] << ", " << c_correct[1] << std::endl;

            // Check if centre calculation was successful
            if (getter::norm(c_correct) == 0.f or getter::norm(x1 - m) == 0.f) {
                // Use vector joining hits
                // 0 and 2 as track direction if 
                // center calculation fails or three
                // hits lie on a straight line
                tangent3D = vector::normalize(x_02);
                // std::cout << "using straight line calculation" << std::endl;
            }

            else {
                // Use circle solution to get track direction

                vector2 r1 = x1 - c_correct;
                vector2 tangent2D{r1[1], -1.f * r1[0]};

                // tangent direction along trajectory
                if (vector::dot(tangent2D, vector2{x_12[0], x_12[1]}) < 0.f)
                    tangent2D = -1.f * tangent2D;
                
                vector2 tangent2D_norm = math::sin(t.m_theta) / getter::norm(tangent2D) * tangent2D;
                
                // track tangent normalized to 1
                tangent3D[0] = tangent2D_norm[0];
                tangent3D[1] = tangent2D_norm[1];
                tangent3D[2] = math::cos(t.m_theta);
            }

            // std::cout << "tangent: " << tangent3D[0] << ", " << tangent3D[1] << ", " << tangent3D[2] << std::endl;


            // Estimate MS-uncertainty
            // (track direction used here
            // to get precise thickness of material)
            
            detray::tracking_surface scat_sf(m_detector, t.m_meas[1].surface_link);

            // effective thickness
            scalar t_eff = mat_scatter / scat_sf.cos_angle({}, tangent3D, t.m_meas[1].local);
            // std::cout << "t_eff " << t_eff << std::endl;

            auto scattering_unc = [](scalar curvature_3D, scalar eff_thickness, vector3 field_strength_vector) {
                return math::fabs(curvature_3D) * 45.f * math::sqrt(eff_thickness) * unit<scalar>::T / field_strength_vector[2] * (1.f + 0.038f * math::log(eff_thickness));
            };

            const auto B_field = m_field.at(t.m_hit_1[0], t.m_hit_1[1], t.m_hit_1[2]);
            vector3 B_vec;
            B_vec[0u] = B_field[0u];
            B_vec[1u] = B_field[1u];
            B_vec[2u] = B_field[2u];
            // std::cout << "\tB-field " << B_vec[0u] << ", " << B_vec[1u] << ", " << B_vec[2u] << std::endl;

            scalar c3D_lin = 0.5f * c_perp * (sin_theta1C + sin_theta2C);
            t.m_sigma_MS = scattering_unc(c3D_lin, t_eff, B_vec);
            // std::cout << "\tsigma_MS " << t.m_sigma_MS << std::endl;

            
            // Index parameters

            scalar n_1C = 1.f / (sin2_theta1C * (0.5f * phi_1C * 1.f / math::tan(0.5f * phi_1C) - 1.f) + 1.f);

            scalar n_2C = 1.f / (sin2_theta2C * (0.5f * phi_2C * 1.f / math::tan(0.5f * phi_2C) - 1.f) + 1.f);

            // Adapt for c_perp = 0
            if (c_perp == 0.f) {
                n_1C = 1.f;
                n_2C = 1.f;
            }

            // Triplet parameters

            // Account for c_perp = 0
            if (c_perp == 0.f) {
                t.m_phi_0 = getter::phi(x_12) - getter::phi(x_01);
                t.m_theta_0 =  theta_2C - theta_1C;
                t.m_rho_phi = -0.5f * getter::norm(x_02);
                t.m_rho_theta = 0.f;

                return;
            }

            t.m_phi_0 = 0.5f * (phi_1C * n_1C + phi_2C * n_2C);
            t.m_theta_0 = theta_2C - theta_1C + (1.f - n_2C) * cos_theta2C / sin_theta2C - (1.f - n_1C) * cos_theta1C / sin_theta1C;
            t.m_rho_phi = -0.5f/c_perp * (phi_1C * n_1C / sin_theta1C + phi_2C * n_2C / sin_theta2C);
            t.m_rho_theta = 1.f/c_perp * ((1.f - n_1C) * cos_theta1C / sin2_theta1C - (1.f - n_2C) * cos_theta2C / sin2_theta2C);

        }

        /// Helper function - Quick Linearize
        ///
        /// Faster calculation of theta_0 & phi_0
        /// using straight line trajectories 
        ///
        /// @param pos0 Position of hit0 (in global frame)
        /// @param pos1 Hit1
        /// @param pos2 Hit2
        ///
        TRACCC_HOST_DEVICE void quick_linearize(const vector3& pos0, const vector3& pos1, const vector3& pos2, scalar& phi_0, scalar& theta_0) {

            // make 2D vector from X, Y components of 3D vector
            auto perp_comp = [](const vector3& vec){ return vector2{vec[0], vec[1]}; };
            
            // Z-component magnitude of x-prod of 2D vectors
            auto cross_2d_z = [](const vector2& v1, const vector2& v2){ return math::fabs(v1[0]*v2[1] - v2[0]*v1[1]); };

            vector2 x_01 = perp_comp(pos1 - pos0);
            vector2 x_12 = perp_comp(pos2 - pos1);

            scalar d_01 = getter::norm(x_01);
            scalar d_12 = getter::norm(x_12);

            // Using cross product
            scalar arg = cross_2d_z(x_01, x_12) / (d_01 * d_12);
            /*std::cout << "x_01 " << x_01[0] << ", " << x_01[1] << std::endl;
            std::cout << "x_12 " << x_12[0] << ", " << x_12[1] << std::endl;
            std::cout << "d_01 " << d_01 << " d_12 " << d_12 << std::endl;*/ 

            phi_0 = math::asin(std::clamp(arg, -1.f, 1.f));
            // std::cout << "arg " << arg << " phi_0 " << phi_0 << std::endl;

            vector2 x_0_L{pos0[2], 0.f};
            vector2 x_1_L{pos1[2], d_01};
            vector2 x_2_L{pos2[2], d_01 + d_12};

            vector2 x_01_L = x_1_L - x_0_L;
            vector2 x_12_L = x_2_L - x_1_L;

            theta_0 = math::asin(cross_2d_z(x_01_L, x_12_L) / (getter::norm(x_01_L) * getter::norm(x_12_L)));
            // std::cout << "\ttheta_0 " << theta_0 << std::endl;
        }

        /// Helper function - Hit Position Derivatives
        ///
        /// Calulation of directional derivatives of
        /// triplet kinks w.r.t hit position shifts
        ///
        /// @param t Triplet
        ///
        TRACCC_HOST_DEVICE void calculate_pos_derivs(triplet& t) {

            // std::cout << "Hit position derivatives:\n";

            scalar phi_0_before = t.m_phi_0;
            scalar theta_0_before = t.m_theta_0;
            scalar phi_0_after;
            scalar theta_0_after;

            // Reserve space for derivative containers
            constexpr size_t max_dims = 2u;
            t.m_h_phi.reserve(3u * max_dims); // hits * max dims/hit
            t.m_h_thet.reserve(3u * max_dims);


            // Loop over measurements in triplet
            for (unsigned hit = 0; hit < 3; ++hit) {

                vector2 pos_loc = t.m_meas[hit].local;
                vector2 var_loc = t.m_meas[hit].variance;

                std::array<vector3, 3u> global_shifted_positions{t.m_hit_0, t.m_hit_1, t.m_hit_2};

                // Surface
                detray::tracking_surface sf(m_detector, t.m_meas[hit].surface_link);

                // over dimensions
                for (unsigned i = 0; i < max_dims; ++i) {

                    // Default derivative 0 for dimensions
                    // which don't exist for this measurement
                    if (i >= t.m_meas[hit].meas_dim) {
                        t.m_h_phi.push_back(0.f);
                        t.m_h_thet.push_back(0.f);
                        continue;
                    }

                    scalar sigma_i = math::sqrt(var_loc[i]);
                    
                    vector2 pos_shifted_loc = pos_loc;

                    // Shift (by the sigma in that direction)
                    pos_shifted_loc[i] = pos_loc[i] + sigma_i;

                    // In global frame
                    vector3 pos_shifted_glob = sf.bound_to_global({}, pos_shifted_loc, {}); 

                    global_shifted_positions[hit] = pos_shifted_glob;

                    // Get parameters with shifted hit
                    quick_linearize(global_shifted_positions[0], global_shifted_positions[1], global_shifted_positions[2], phi_0_after, theta_0_after);

                    t.m_h_phi.push_back((phi_0_after - phi_0_before) / sigma_i);
                    t.m_h_thet.push_back((theta_0_after - theta_0_before) / sigma_i);
                }

            }


            /*
            // Print derivatives
            std::cout << "\tH_theta: ";
            for (unsigned j = 0; j < t.m_h_thet.size(); ++j) {
                std::cout << " " << t.m_h_thet[j];
            }
            std::cout << "\n\tH_phi: ";
            for (unsigned j = 0; j < t.m_h_phi.size(); ++j) {
                std::cout << " " << t.m_h_phi[j];
            }
            */

        }

        /// Helper function - Global Fit
        ///
        /// Global fit of hit triplets on track
        ///
        /// @param fitting_res result of the fit for this track
        /// @param track_states fitted track states at the measurement surfaces
        ///
        /// (fitted state only at the first measurement surface is calculated)
        TRACCC_HOST_DEVICE
        void do_global_fit(fitting_result<algebra_type>& fitting_res, 
            vector_type<track_state<algebra_type>>& track_states) {

            // Allocate matrices with max possible sizes
            constexpr size_t max_dims = 2u;
            constexpr size_t max_nhits = 20u; // Assumption about max number of hits
            constexpr size_t max_ntrips = max_nhits - 2u;
            constexpr size_t max_ndirs = max_dims * max_nhits;

            // Actual number in this track
            const size_t N_triplets = m_triplets.size();
            assert(m_track_states.size() <= max_nhits); 
            assert(N_triplets == m_track_states.size() - 2u);


            // Make matrices/vectors

            // Triplet parameter vectors
            matrix_type<2u * max_ntrips, 1u> rho = matrix_operator().template zero<2u * max_ntrips, 1u>();
            matrix_type<2u * max_ntrips, 1u> psi = matrix_operator().template zero<2u * max_ntrips, 1u>();
            
            // Scattering & hit precision matrices
            // (directly the covariance matrices as 
            // D_hit or D_MS are never used as they are)
            matrix_type<2u * max_ntrips, 2u * max_ntrips> D_MS_inv = matrix_operator().template identity<2u * max_ntrips, 2u * max_ntrips>();
            matrix_type<max_ndirs, max_ndirs> D_hit_inv = matrix_operator().template zero<max_ndirs, max_ndirs>();

            // Hit gradient (Jacobian) matrix
            matrix_type<2u*max_ntrips, max_ndirs> H = matrix_operator().template zero<2u*max_ntrips, max_ndirs>();
            
            // Fill matrices/vectors

            for (size_t i = 0; i < N_triplets; ++i) {

                const triplet& t_i = m_triplets[i];

                getter::element(rho, i, 0u) = t_i.m_rho_theta;
                getter::element(rho, i + N_triplets, 0u) = t_i.m_rho_phi;

                getter::element(psi, i, 0u) = t_i.m_theta_0;
                getter::element(psi, i + N_triplets, 0u) = t_i.m_phi_0;

                // Only update elements when linearization
                // has been done for this triplet
                // (reject on default value) 
                if (t_i.m_sigma_MS != 0.f) {
                    scalar sigma2_MS = t_i.m_sigma_MS * t_i.m_sigma_MS;
                    scalar sin2_theta = math::sin(t_i.m_theta);
                    sin2_theta *= sin2_theta;
                    getter::element(D_MS_inv, i, i) = sigma2_MS;
                    getter::element(D_MS_inv, i + N_triplets, i + N_triplets) = sigma2_MS / sin2_theta;
                }


                // (after unrolling loop over hits & uncertainty dimensions)
                // 1st Hit in triplet
                getter::element(H, i, i) = t_i.m_h_thet[0u];
                getter::element(H, i, max_nhits + i) = t_i.m_h_thet[1u];
                // 2nd Hit
                getter::element(H, i, i + 1u) = t_i.m_h_thet[max_dims*1u];
                getter::element(H, i, max_nhits + i + 1u) = t_i.m_h_thet[max_dims*1u + 1u];
                // 3rd Hit
                getter::element(H, i, i + 2u) = t_i.m_h_thet[max_dims*2u];
                getter::element(H, i, max_nhits + i + 2u) = t_i.m_h_thet[max_dims*2u + 1u];

                // 1st Hit
                getter::element(H, i + N_triplets, i) = t_i.m_h_phi[0u];
                getter::element(H, i + N_triplets, max_nhits + i) = t_i.m_h_phi[1u];
                // 2nd Hit
                getter::element(H, i + N_triplets, i + 1u) = t_i.m_h_phi[max_dims*1u];
                getter::element(H, i + N_triplets, max_nhits + i + 1u) = t_i.m_h_phi[max_dims*1u + 1u];
                // 3rd Hit
                getter::element(H, i + N_triplets, i + 2u) = t_i.m_h_phi[max_dims*2u];
                getter::element(H, i + N_triplets, max_nhits + i + 2u) = t_i.m_h_phi[max_dims*2u + 1u];
            

                // 1st Hit in triplet
                getter::element(D_hit_inv, i, i) = t_i.m_meas[0u].variance[0u];
                getter::element(D_hit_inv, i + max_nhits, i + max_nhits) = t_i.m_meas[0u].variance[1u];

                // Only use the other two hits
                // for the last triplet to prevent
                // reassigning elements in matrix
                if (i == N_triplets - 1u) {
                    // 2nd hit
                    getter::element(D_hit_inv, i + 1u, i + 1u) = t_i.m_meas[1u].variance[0u];
                    getter::element(D_hit_inv, i + max_nhits + 1u, i + max_nhits + 1u) = t_i.m_meas[1u].variance[1u];

                    // 3rd hit
                    getter::element(D_hit_inv, i + 2u, i + 2u) = t_i.m_meas[2u].variance[0u];
                    getter::element(D_hit_inv, i + max_nhits + 2u, i + max_nhits + 2u) = t_i.m_meas[2u].variance[1u];
                }

            } // done filling



            // Triplet precision matrix
            // Note: diagonal elements in K_inv are 1
            // corresponding to unused 'objects', since
            // the same is true in D_MS_inv and those in
            // H * D_hit^-1 * H^T are 0

            matrix_type<2u*max_ntrips, 2u*max_ntrips> K_inv = D_MS_inv + H * D_hit_inv * matrix_operator().transpose(H);

            // Matrix inversion
            matrix_type<2u*max_ntrips, 2u*max_ntrips> K = matrix_operator().inverse(K_inv);    

            /*
            std::cout << " ************************************ MATRICES ************************************ " << std::endl;
            std::cout << "D_MS_inv:\n";
            for (size_t r = 0u; r < 2u*max_ntrips; ++r) {
                for (size_t c = 0u; c < 2u*max_ntrips; ++c) {
                    std::cout << std::setw(12);
                    std::cout << getter::element(D_MS_inv, r, c) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "D_hit_inv:\n";
            for (size_t r = 0u; r < max_ndirs; ++r) {
                for (size_t c = 0u; c < max_ndirs; ++c) {
                    std::cout << std::setw(12);
                    std::cout << getter::element(D_hit_inv, r, c) << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "K_inv:\n";
            for (size_t r = 0u; r < 2u*max_ntrips; ++r) {
                for (size_t c = 0u; c < 2u*max_ntrips; ++c) {
                    std::cout << std::setw(12);
                    std::cout << getter::element(K_inv, r, c) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "K:\n";
            for (size_t r = 0u; r < 2u*max_ntrips; ++r) {
                for (size_t c = 0u; c < 2u*max_ntrips; ++c) {
                    std::cout << std::setw(12);
                    std::cout << getter::element(K, r, c) << " ";
                }
                std::cout << std::endl;
            }*/

            matrix_type<1u, 1u> num = -1.f * matrix_operator().transpose(rho) * K * psi;
            matrix_type<1u, 1u> den = matrix_operator().transpose(rho) * K * rho;
            matrix_type<1u, 1u> psiT_K_psi = matrix_operator().transpose(psi) * K * psi;

            // Calculation of curvature, uncertainty, fit quality

            scalar c_3D = getter::element(num, 0u, 0u) / getter::element(den, 0u, 0u);

            // scalar sigma_c_3D = 1.f / math::sqrt(getter::element(den, 0u, 0u)); // not used now

            scalar chi2 = getter::element(psiT_K_psi, 0u, 0u) - (getter::element(num, 0u, 0u) * getter::element(num, 0u, 0u)) / getter::element(den, 0u, 0u); 

            // std::cout << "\nGlobal fit: c_3D " << c_3D << "  sigma_c_3D " << sigma_c_3D << "  chi2 " << chi2 << std::endl;

            // Calculation of hit residuals

            matrix_type<2u*max_ntrips, 2u*max_ntrips> K_rho = K - (1.f / getter::element(den, 0u, 0u)) * K * rho * matrix_operator().transpose(rho) * K;

            matrix_type<max_ndirs, 1u> delta_fit = D_hit_inv * matrix_operator().transpose(H) * K_rho * psi;

            // std::cout << "posn. shift hit 0: " << getter::element(delta_fit, 0u, 0u) << " " << getter::element(delta_fit, 1u, 0u) << " " << getter::element(delta_fit, 2u, 0u) << std::endl; 


            // Track parameters at the first measurement surface
            auto fitted_params = [&delta_fit, &c_3D](
                const vector_type<track_state<algebra_type>>& input_states,
                const vector_type<triplet>& triplets,
                const detector_t& detector,
                const bfield_t& field) -> detray::bound_parameters_vector<algebra_type> {
                
                // Get the (post-fit) global positions of 
                // the first and the second measurement

                // First measurement
                auto m0 = input_states[0].get_measurement();
                
                point2 loc0{m0.local[0], m0.local[1]};
                point2 loc0_post_fit = loc0 + point2{getter::element(delta_fit, 0u, 0u), getter::element(delta_fit, max_nhits, 0u)};

                detray::tracking_surface sf0(detector, m0.surface_link);

                point3 glob0 = sf0.bound_to_global({}, loc0_post_fit, {});

                // Second measurement
                auto m1 = input_states[1].get_measurement();

                point2 loc1{m1.local[0], m1.local[1]};
                point2 loc1_post_fit = loc1 + point2{getter::element(delta_fit, 1u, 0u), getter::element(delta_fit, 1u + max_nhits, 0u)};

                detray::tracking_surface sf1(detector, m1.surface_link);

                point3 glob1 = sf1.bound_to_global({}, loc1_post_fit, {});

                point3 r01 = glob1 - glob0;

                // Calculation of track parameters at the first
                // measurement with the first two hits assuming
                // small bending

                scalar bending_angle = c_3D * getter::norm(r01);

                // Magnetic field at first measurement
                const auto B_field = field.at(triplets[0].m_hit_0[0u], triplets[0].m_hit_0[1u], triplets[0].m_hit_0[2u]);
                vector3 B_vec;
                B_vec[0u] = B_field[0u];
                B_vec[1u] = B_field[1u];
                B_vec[2u] = B_field[2u];

                // Momentum - TODO: handling of magnetic field
                // Units: B [T], p [MeV] 
                scalar p = 0.3f * getter::norm(B_vec) / (c_3D * unit<scalar>::T) * unit<scalar>::MeV; 

                // Wrap angle between -Pi and Pi
                auto wrap_pi_mpi = [](scalar angle) -> scalar {
                    
                    if (angle > static_cast<scalar>(M_PI))
                        return angle - 2.f * static_cast<scalar>(M_PI);
                    
                    else if (angle < -1.f * static_cast<scalar>(M_PI))
                        return angle + 2.f * static_cast<scalar>(M_PI);
                    
                    else
                        return angle;
                };
                
                // Set parameters
                const scalar q = 1.f;
                detray::bound_parameters_vector<algebra_type> params_vec{};
                params_vec.set_bound_local(loc0_post_fit);

                // std::cout << "phi r01 " << getter::phi(r01) << " bending angle " << bending_angle << std::endl;
                // std::cout << "phi " << getter::phi(r01) + 0.5f * bending_angle << " wrapped phi " << wrap_pi_mpi(getter::phi(r01) + 0.5f * bending_angle) << std::endl;

                params_vec.set_phi(wrap_pi_mpi(getter::phi(r01) + 0.5f * bending_angle));
                scalar theta = math::atan2(getter::perp(r01) * 0.5f * bending_angle, r01[2u] * math::sin(0.5f * bending_angle));
                // std::cout << getter::theta(r01) << " " << theta << std::endl;
                params_vec.set_theta(math::fabs(theta));
                params_vec.set_qop(q / p);
                params_vec.set_time(0.f);

                // std::cout << "p " << p << std::endl;
                
                return params_vec;

            }(m_track_states, m_triplets, m_detector, m_field);

            fitting_res.chi2 = chi2;
            fitting_res.fit_params.set_vector(fitted_params.vector());
            fitting_res.ndf = [](const vector_type<track_state<algebra_type>>& states)
            -> scalar {
                
                // Number of degrees of freedom
                // = sum of number of dimensions
                // of measurements - number of
                // track parameter dimensions.

                scalar sum_dims = 0;
                for (const track_state<algebra_type>& s : states) {
                    sum_dims += static_cast<scalar>(s.get_measurement().meas_dim);
                }

                return (sum_dims - 5.f);
            }(m_track_states);

            // Only the smoothed parameters
            // at the first measurement are 
            // used for performance plots
            // see fitting_performance_writer
            track_states[0].smoothed_chi2() = chi2;
            track_states[0].smoothed().set_vector(fitted_params.vector());
            track_states[0].is_hole = false;
            
        }
        
        /// Run the fitter
        ///
        /// Main fitting function
        /// 
        /// @param fitting_res result of the fit for this track
        /// @param track_states fitted track states at the measurement surfaces
        /// (fitted state only at the first measurement surface is calculated)
        TRACCC_HOST_DEVICE void fit(fitting_result<algebra_type>& fitting_res, 
            vector_type<track_state<algebra_type>>& track_states) {

            // std::cout << "Fitting track with " << m_triplets.size() << " triplets\n";

            unsigned triplet_idx = 0;

            for (triplet& t : m_triplets) {
                
                // std::cout << "Triplet " << triplet_idx << "\n";
                
                linearize_triplet(t);
                calculate_pos_derivs(t);
                
                ++triplet_idx;
                
            }
            
            // Passing through the input track
            // states (measurements) to the output
            for (const auto& state : m_track_states) {
                track_states.push_back(state);
            }

            // Only the smoothed parameter
            // at the first measurement is
            // updated here in the vector.
            do_global_fit(fitting_res, track_states);

        }
        
        
        private:

        // Hard-coded material
        // TODO: get from surface after re-mapping
        scalar mat_scatter = 0.01f;

        // Detector context type
        using context = typename detector_t::geometry_context;

        // Detector object
        const detector_t& m_detector;
        // Field object
        const bfield_t m_field;
        
        // Vector of triplets
        vector_type<triplet> m_triplets;
        // Track states
        vector_type<track_state<algebra_type>> m_track_states;
        
        // Configuration object
        config_type m_cfg;
    
    };

} // namespace traccc
