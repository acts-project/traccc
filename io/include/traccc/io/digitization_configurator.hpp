/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"

// Project include(s).
#include "traccc/io/digitization_config.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"
#include "detray/geometry/surface.hpp"

// System include(s).
#include <string_view>

namespace Acts {
class BinUtility;
}

namespace traccc::io {

/// Helper configurator that takes a simplified (per volume, per extra bit)
/// input digitization file and creates a full fletched per module
/// digitization configuration file.
///
/// It acts as a visitor and then builds a fully developed digitization file
/// for the geometric digitization, filling in the correct dimensions and
/// number of bins.
///
/// The simplified file is assumed to have just one bin for the geometric
/// digitization, which is then used to calculate the number of bins with
/// respect to the bounds range.
///
/// @see
/// https://github.com/acts-project/acts/blob/main/Examples/Algorithms/Digitization/include/ActsExamples/Digitization/DigitizationConfigurator.hpp
struct digitization_configurator {
    /// Simplified input components for digitization (meta-configuration)
    digitization_config input_digi_cfg;

    /// Final collection of output components
    std::vector<
        std::pair<detray::geometry::barcode, module_digitization_config>>
        output_digi_cfg;

    /// Needs an input configuration
    digitization_configurator() = delete;

    /// Construct from an input configuration @param cfg and initializes an
    /// empty output configuration
    digitization_configurator(digitization_config cfg)
        : input_digi_cfg{cfg}, output_digi_cfg{} {}

    /// The visitor call for the geometry
    ///
    /// @param surface is the surfaces that is visited
    ///
    /// Takes the @c input_digi_cfg and adds an appropriate entry into the
    /// @c output_digi_cfg for the given surface
    template <class detector_t>
    void operator()(const detray::surface<detector_t> &surface);

    struct segmentation_configurator {

        /// Create the segmentation for a specific surface from the
        /// given configuration in @param input_segmentation.
        template <typename mask_t>
        void fill_output_segmentation(
            const mask_t &mask, const Acts::BinUtility &input_segmentation,
            Acts::BinUtility &output_segmentation) const;

        /// Visitor for the surface mask
        template <typename mask_group_t, typename index_t>
        inline void operator()(const mask_group_t &mask_group,
                               const index_t &index,
                               const Acts::BinUtility &input_segmentation,
                               Acts::BinUtility &output_segmentation) const {

            fill_output_segmentation(mask_group[index], input_segmentation,
                                     output_segmentation);
        }
    };
};

}  // namespace traccc::io

#include "traccc/io/digitization_configurator.ipp"
