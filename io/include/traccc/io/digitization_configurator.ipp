/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray include(s)
#include "detray/masks/masks.hpp"

// Acts include(s).
#include <Acts/Geometry/GeometryIdentifier.hpp>
#include <Acts/Utilities/BinUtility.hpp>

// System include(s)
#include <algorithm>
#include <cmath>

namespace traccc::io {

template <class detector_t>
void digitization_configurator::operator()(
    const detray::surface<detector_t> &surface) {

    if (!surface.is_sensitive()) {
        return;
    }

    // For geometries that are generated in detray, there is no source link
    Acts::GeometryIdentifier geo_id;
    if (surface.source() ==
        std::numeric_limits<decltype(surface.source())>::max()) {
        geo_id = Acts::GeometryIdentifier(0);
        geo_id.setVolume(surface.volume());
        // Translate detray broadcast value to acts broadcast value
        geo_id.setExtra(surface.extra() == 255 ? 0 : surface.extra());
    } else {
        geo_id = Acts::GeometryIdentifier(surface.source());
    }

    auto input_cfg = input_digi_cfg.find(geo_id);

    if (input_cfg != input_digi_cfg.end()) {
        // The output config, copy over the smearing part
        module_digitization_config output_cfg;

        if (!input_cfg->indices.empty()) {
            // Copy the measurement indices
            output_cfg.indices = input_cfg->indices;

            // Create the output segmentation
            surface.template visit_mask<segmentation_configurator>(
                input_cfg->segmentation, output_cfg.segmentation);
        }

        // Insert into the output list
        output_digi_cfg.push_back({surface.barcode(), output_cfg});
    }
}

template <class mask_t>
void digitization_configurator::segmentation_configurator::
    fill_output_segmentation(const mask_t &mask,
                             const Acts::BinUtility &input_segmentation,
                             Acts::BinUtility &output_segmentation) const {

    using bounds = typename mask_t::boundaries;
    const auto &bound_values = mask.values();

    [[maybe_unused]] unsigned int accessBin{
        (input_segmentation.dimensions() == 2) ? 1u : 0u};

    // The module is a rectangular module
    if constexpr (std::is_same_v<typename mask_t::shape,
                                 detray::rectangle2D<>>) {
        if (input_segmentation.binningData()[0].binvalue == Acts::binX) {

            auto minX{-bound_values[bounds::e_half_x]};
            auto maxX{bound_values[bounds::e_half_x]};
            auto n_bins{static_cast<unsigned int>(std::round(
                (maxX - minX) / input_segmentation.binningData()[0].step))};

            output_segmentation +=
                Acts::BinUtility(n_bins, minX, maxX, Acts::open, Acts::binX);
        }
        if (input_segmentation.binningData()[0].binvalue == Acts::binY or
            input_segmentation.dimensions() == 2) {

            auto minY{-bound_values[bounds::e_half_y]};
            auto maxY{bound_values[bounds::e_half_y]};
            auto n_bins{static_cast<unsigned int>(
                std::round((maxY - minY) /
                           input_segmentation.binningData()[accessBin].step))};

            output_segmentation +=
                Acts::BinUtility(n_bins, minY, maxY, Acts::open, Acts::binY);
        }
    }
    // The module is a trapezoid module
    else if constexpr (std::is_same_v<typename mask_t::shape,
                                      detray::trapezoid2D<>>) {

        if (input_segmentation.binningData()[0].binvalue == Acts::binX) {

            auto maxX{std::max(bound_values[bounds::e_half_length_0],
                               bound_values[bounds::e_half_length_1])};
            auto n_bins{static_cast<unsigned int>(std::round(
                2 * maxX / input_segmentation.binningData()[0].step))};

            output_segmentation +=
                Acts::BinUtility(n_bins, -maxX, maxX, Acts::open, Acts::binX);
        }
        if (input_segmentation.binningData()[0].binvalue == Acts::binY or
            input_segmentation.dimensions() == 2) {

            auto maxY{bound_values[bounds::e_half_length_2]};
            auto n_bins{static_cast<unsigned int>(
                std::round((2 * maxY) /
                           input_segmentation.binningData()[accessBin].step))};

            output_segmentation +=
                Acts::BinUtility(n_bins, -maxY, maxY, Acts::open, Acts::binY);
        }
    }
    // The module is an annulus module
    else if constexpr (std::is_same_v<typename mask_t::shape,
                                      detray::annulus2D<>>) {

        if (input_segmentation.binningData()[0].binvalue == Acts::binR) {

            auto minR{bound_values[bounds::e_min_r]};
            auto maxR{bound_values[bounds::e_max_r]};
            auto n_bins{static_cast<unsigned int>(std::round(
                (maxR - minR) / input_segmentation.binningData()[0].step))};

            output_segmentation +=
                Acts::BinUtility(n_bins, minR, maxR, Acts::open, Acts::binR);
        }
        if (input_segmentation.binningData()[0].binvalue == Acts::binPhi or
            input_segmentation.dimensions() == 2) {

            auto averagePhi{bound_values[bounds::e_average_phi]};
            auto minPhi{averagePhi + bound_values[bounds::e_min_phi_rel]};
            auto maxPhi{averagePhi + bound_values[bounds::e_max_phi_rel]};
            auto n_bins{static_cast<unsigned int>(
                std::round((maxPhi - minPhi) /
                           input_segmentation.binningData()[accessBin].step))};

            output_segmentation += Acts::BinUtility(n_bins, minPhi, maxPhi,
                                                    Acts::open, Acts::binPhi);
        }
    }
}

}  // namespace traccc::io
