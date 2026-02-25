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
#include <set>

namespace {

void fill_digi_info(traccc::detector_module_description::host& dmd,
                    const traccc::module_digitization_config& data) {


    dmd.dimensions().back() = data.dimensions;

    dmd.bin_edges_x().back().assign(
        data.bin_edges[0].begin(), data.bin_edges[0].end());

    dmd.bin_edges_y().back().assign(
        data.bin_edges[1].begin(), data.bin_edges[1].end());

}

template <typename detector_traits_t>
void read_json_dd_impl(traccc::detector_module_description::host& dmd,
                       traccc::detector_conditions_description::host& dcd,
                       const traccc::host_detector& detector,
                       const traccc::digitization_config& digi)
requires(traccc::is_detector_traits<detector_traits_t>)
{
    const typename detector_traits_t::host& detector_host =
        detector.as<detector_traits_t>();


    dmd.reserve(digi.designs.size());
    dcd.reserve(detector_host.surfaces().size());

    std::unordered_map<uint64_t, int> id_to_design_index = digi.id_to_design_index;
    std::set<int> design_indices;

    std::unordered_map<int, unsigned int> design_index_to_dd_pos;

    // Will be populated in surface order — one entry per sensitive surface
    std::vector<unsigned int> module_to_design;

    for (const auto& surface_desc : detector_host.surfaces()) {
        const traccc::geometry_id geom_id{surface_desc.source};
        const Acts::GeometryIdentifier acts_geom_id{geom_id};

        if (acts_geom_id.sensitive() == 0) {
            continue;
        }

        dcd.geometry_id().back() = surface_desc.barcode();
        dcd.acts_geometry_id().back() = geom_id;
        dcd.threshold().back() = ;
        dcd.measurement_translation().back() = ;

        int design_index = id_to_design_index[surface_desc.barcode().value()];

        if (design_indices.contains(design_index)) {
            // Design already added — just record the index
            module_to_design.push_back(design_index_to_dd_pos[design_index]);

        } else {
            // New design — add it to dd
            dd.resize(dd.size() + 1);
            design_indices.insert(design_index);

            const traccc::module_digitization_config* digi_cfg =
                digi.get(surface_desc.barcode().value());

            if (digi_cfg == nullptr) {
                std::ostringstream msg;
                msg << "Could not find digitization config for barcode: "
                    << surface_desc.barcode();
                throw std::runtime_error(msg.str());
            }

            // The position this design will occupy in dd
            const unsigned int dd_pos =
                static_cast<unsigned int>(design_index_to_dd_pos.size());
            design_index_to_dd_pos[design_index] = dd_pos;

            dd.subspace().back() = {0, 1};
            using annulus_t =
                detray::mask<detray::annulus2D, traccc::default_algebra>;
            using mask_registry_t = typename detector_traits_t::host::masks;
            if constexpr (detray::types::contains<mask_registry_t, annulus_t>) {
                if (surface_desc.mask().id() ==
                    detray::types::id<mask_registry_t, annulus_t>) {
                    dd.subspace().back() = {1, 0};
                }
            }

            fill_digi_info(dd, *digi_cfg);

            // Record the indirection for this module too
            module_to_design.push_back(dd_pos);
        }
    }

    dd.module_to_design().assign(module_to_design.begin(), module_to_design.end());

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
