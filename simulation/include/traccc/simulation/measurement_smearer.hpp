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
#include "detray/geometry/shapes/annulus2D.hpp"
#include "detray/geometry/shapes/line.hpp"
#include "detray/tracks/bound_track_parameters.hpp"

// System include(s).
#include <array>
#include <random>
#include <string>

namespace traccc {

template <typename algebra_t>
struct measurement_smearer {

    using algebra_type = algebra_t;
    using matrix_operator = detray::dmatrix_operator<algebra_t>;
    using scalar_type = detray::dscalar<algebra_t>;
    template <std::size_t ROWS, std::size_t COLS>
    using matrix_type = detray::dmatrix<algebra_t, ROWS, COLS>;

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
        const detray::bound_track_parameters<algebra_t>& bound_params,
        io::csv::measurement& iomeas) {

        // Line detector
        if constexpr (std::is_same_v<typename mask_t::local_frame_type,
                                     detray::line2D<traccc::default_algebra>>) {
            iomeas.local_key = 2;
        }
        // Annulus strip
        else if constexpr (std::is_same_v<typename mask_t::shape,
                                          detray::annulus2D>) {
            iomeas.local_key = 4;
        }
        // Else
        else {
            iomeas.local_key = 6;
        }

        std::array<detray::dsize_type<algebra_t>, 2u> indices{0u, 0u};
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

        subspace<traccc::default_algebra, 6u, 2u> subs(indices);

        if (meas_dim == 1u) {
            const auto proj = subs.projector<1u>();
            matrix_type<1u, 1u> meas = proj * bound_params.vector();

            if constexpr (std::is_same_v<
                              typename mask_t::local_frame_type,
                              detray::line2D<traccc::default_algebra>>) {
                iomeas.local0 =
                    std::max(std::abs(matrix_operator().element(meas, 0u, 0u)) +
                                 offset[0],
                             static_cast<scalar_type>(0.f));
            } else if constexpr (std::is_same_v<typename mask_t::shape,
                                                detray::annulus2D>) {
                iomeas.local1 =
                    matrix_operator().element(meas, 0u, 0u) + offset[0];
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
