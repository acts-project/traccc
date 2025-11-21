/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/io/read_detector_description.hpp"

#include "traccc/geometry/host_detector.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/utils.hpp"

// Detray include(s)
#include <detray/geometry/tracking_surface.hpp>
#include <detray/io/frontend/impl/json_readers.hpp>
#include <detray/utils/type_registry.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <sstream>
#include <stdexcept>

namespace {

void fill_digi_info(traccc::silicon_detector_description::host& dd,
                    const traccc::module_digitization_config& data) {

    // Set a hard-coded threshold for which cells should be considered for
    // clusterization on this module / surface.
    dd.threshold().back() = 0.f;

    // Fill the new element with the digitization configuration for the
    // surface.
    const auto& binning_data = data.segmentation.binningData();
    dd.reference_x().back() = binning_data.at(0).min;
    dd.reference_y().back() = binning_data.at(1).min;
    dd.pitch_x().back() = binning_data.at(0).step;
    dd.pitch_y().back() = binning_data.at(1).step;
    dd.dimensions().back() = data.dimensions;
}

template <typename detector_traits_t>
void read_json_dd_impl(traccc::silicon_detector_description::host& dd,
                       const traccc::host_detector& detector,
                       const traccc::digitization_config& digi)
    requires(traccc::is_detector_traits<detector_traits_t>)
{
    const typename detector_traits_t::host& detector_host =
        detector.as<detector_traits_t>();

    // Iterate over the surfaces of the detector.
    const typename detector_traits_t::host::surface_lookup_container& surfaces =
        detector_host.surfaces();
    dd.reserve(surfaces.size());
    for (const auto& surface_desc : detector_host.surfaces()) {

        // Acts geometry identifier(s) for the surface.
        const traccc::geometry_id geom_id{surface_desc.source};
        const Acts::GeometryIdentifier acts_geom_id{geom_id};

        // Skip non-sensitive surfaces. They are not needed in the
        // "detector description".
        if (acts_geom_id.sensitive() == 0) {
            continue;
        }

        // Add a new element to the detector description.
        dd.resize(dd.size() + 1);

        // Construct a Detray surface object.
        const detray::tracking_surface surface{detector_host, surface_desc};

        // Fill the new element with the geometry ID and the transformation of
        // the surface in question.
        dd.geometry_id().back() = surface_desc.barcode();
        dd.acts_geometry_id().back() = geom_id;
        dd.measurement_translation().back() = {0.f, 0.f};
        dd.subspace().back() = {0, 1};

        // ITk specific type
        using annulus_t =
            detray::mask<detray::annulus2D, traccc::default_algebra>;
        using mask_registry_t = typename detector_traits_t::host::masks;
        if constexpr (detray::types::contains<mask_registry_t, annulus_t>) {
            if (surface_desc.mask().id() ==
                detray::types::id<mask_registry_t, annulus_t>) {
                dd.subspace().back() = {1, 0};
            }
        }

        // Find the module's digitization configuration.
        const traccc::digitization_config::Iterator digi_it =
            digi.find(acts_geom_id);
        if (digi_it == digi.end()) {
            std::ostringstream msg;
            msg << "Could not find digitization config for geometry ID: "
                << acts_geom_id;
            throw std::runtime_error(msg.str());
        }

        // Fill the new element with the digitization configuration for the
        // surface.
        fill_digi_info(dd, *digi_it);
    }
}

void read_json_dd(traccc::silicon_detector_description::host& dd,
                  std::string_view geometry_file,
                  const traccc::digitization_config& digi) {

    // Construct a (temporary) Detray detector object from the geometry
    // configuration file.
    vecmem::host_memory_resource mr;
    traccc::host_detector detector;
    traccc::io::read_detector(detector, mr, geometry_file);

    // TODO: Implement detector visitor!
    // Peek at the header to determine the kind of detector that is needed
    const auto header = detray::io::detail::deserialize_json_header(
        traccc::io::get_absolute_path(geometry_file));

    if (header.detector == "Cylindrical detector from DD4hep blueprint") {
        read_json_dd_impl<traccc::odd_detector>(dd, detector, digi);
    } else if (header.detector == "detray_detector") {
        read_json_dd_impl<traccc::itk_detector>(dd, detector, digi);
    } else {
        // TODO: Warning here
        read_json_dd_impl<traccc::default_detector>(dd, detector, digi);
    }
}

}  // namespace

namespace traccc::io {

void read_detector_description(silicon_detector_description::host& dd,
                               std::string_view geometry_file,
                               std::string_view digitization_file,
                               const data_format geometry_format,
                               const data_format digitization_format) {

    // Read the digitization configuration.
    const digitization_config digi =
        read_digitization_config(digitization_file, digitization_format);

    // Fill the detector description with the correct type of geometry file.
    switch (geometry_format) {
        case data_format::json:
            ::read_json_dd(dd, geometry_file, digi);
            break;
        default:
            throw std::invalid_argument("Unsupported geometry format.");
    }
}

}  // namespace traccc::io
