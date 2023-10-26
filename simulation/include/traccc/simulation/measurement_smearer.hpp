/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/io/csv/measurement.hpp"
#include "traccc/utils/subspace.hpp"

// Detray include(s).
#include "detray/tracks/bound_track_parameters.hpp"

// System include(s).
#include <array>
#include <random>
#include <string>

namespace traccc {

template <typename transform3_t>
struct measurement_smearer {

    using transform3_type = transform3_t;
    using matrix_operator = typename transform3_t::matrix_actor;
    using scalar_type = typename transform3_t::scalar_type;
    using size_type = typename matrix_operator::size_ty;
    template <size_type ROWS, size_type COLS>
    using matrix_type =
        typename matrix_operator::template matrix_type<ROWS, COLS>;

    measurement_smearer(const scalar_type stddev_local0,
                        const scalar_type stddev_local1)
        : stddev({stddev_local0, stddev_local1}) {}

    measurement_smearer(measurement_smearer& smearer)
        : stddev(smearer.stddev), generator(smearer.generator) {}

    void set_seed(const uint_fast64_t sd) { generator.seed(sd); }

    std::array<scalar_type, 2> stddev;
    std::random_device rd{};
    std::mt19937_64 generator{rd()};

    std::array<scalar_type, 2> get_offset() {
        return {
            std::normal_distribution<scalar_type>(0.f, stddev[0])(generator),
            std::normal_distribution<scalar_type>(0.f, stddev[1])(generator)};
    }

    template <typename mask_t>
    void operator()(
        const mask_t& /*mask*/, const std::array<scalar_type, 2>& offset,
        const detray::bound_track_parameters<transform3_t>& bound_params,
        io::csv::measurement& iomeas) {

        // Line detector
        if (mask_t::shape::name == "line") {
            iomeas.local_key = 2;
        }
        // Annulus strip
        else if (mask_t::shape::name == "(stereo) annulus2D") {
            iomeas.local_key = 4;
        }
        // Else
        else {
            iomeas.local_key = 6;
        }

        std::array<typename transform3::size_type, 2u> indices{0u, 0u};
        unsigned int meas_dim = 0u;
        for (unsigned int ipar = 0; ipar < 2u; ++ipar) {
            if (((iomeas.local_key) & (1 << (ipar + 1))) != 0) {
                switch (ipar) {
                    case e_bound_loc0: {
                        indices[meas_dim++] = ipar;
                    }; break;
                    case e_bound_loc1: {
                        indices[meas_dim++] = ipar;
                    }; break;
                }
            }
        }

        subspace<transform3, 6u, 2u> subs(indices);

        if (meas_dim == 1u) {
            const auto proj = subs.projector<1u>();
            matrix_type<1u, 1u> meas = proj * bound_params.vector();

            if (mask_t::shape::name == "line") {
                iomeas.local0 =
                    std::max(std::abs(matrix_operator().element(meas, 0u, 0u)) +
                                 offset[0],
                             static_cast<scalar_type>(0.f));
            } else {
                iomeas.local0 =
                    matrix_operator().element(meas, 0u, 0u) + offset[0];
            }

        } else if (meas_dim == 2u) {
            const auto proj = subs.projector<2u>();
            matrix_type<2u, 1u> meas = proj * bound_params.vector();

            iomeas.local0 = matrix_operator().element(meas, 0u, 0u) + offset[0];
            iomeas.local1 = matrix_operator().element(meas, 1u, 0u) + offset[1];
        }

        return;
    }
};

}  // namespace traccc
