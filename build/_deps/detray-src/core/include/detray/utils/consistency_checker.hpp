/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/geometry/tracking_surface.hpp"
#include "detray/geometry/tracking_volume.hpp"
#include "detray/materials/detail/concepts.hpp"
#include "detray/materials/predefined_materials.hpp"
#include "detray/utils/ranges.hpp"

// System include(s)
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace detray::detail {

/// @returns a string that identifies the volume
template <typename detector_t>
std::string print_volume_name(const tracking_volume<detector_t> &vol,
                              const typename detector_t::name_map &names) {
    if (names.empty()) {
        return std::to_string(vol.index());
    } else {
        return vol.name(names);
    }
}

/// Checks every collection in a multistore to be emtpy and prints a warning
template <typename store_t, std::size_t... I>
void report_empty(const store_t &store, const std::string &store_name,
                  std::index_sequence<I...> /*seq*/) {

    ((store.template empty<store_t::value_types::to_id(I)>()
          ? std::cout << "WARNING: " << store_name
                      << " has empty collection no. " << I << std::endl
          : std::cout << ""),
     ...);
}

/// A functor that checks the surface descriptor and correct volume index in
/// every acceleration data structure for a given volume
struct surface_checker {

    /// Test the contained surfaces for consistency
    template <typename detector_t>
    DETRAY_HOST_DEVICE void operator()(
        const typename detector_t::surface_type &sf_descr,
        const detector_t &det, const dindex vol_idx,
        const typename detector_t::name_map &names) const {

        const auto sf = tracking_surface{det, sf_descr};
        const auto vol = tracking_volume{det, vol_idx};
        std::stringstream err_stream{};
        err_stream << "VOLUME \"" << print_volume_name(vol, names) << "\":\n";

        if (!sf.self_check(err_stream)) {
            throw std::invalid_argument(err_stream.str());
        }

        if (sf.volume() != vol_idx) {
            err_stream << "ERROR: Incorrect volume index on surface: vol "
                       << vol_idx << ", sf: " << sf;

            throw std::invalid_argument(err_stream.str());
        }

        // Does the mask link to an existing volume?
        if (!detail::is_invalid_value(sf.volume_link()) &&
            (sf.volume_link() >= det.volumes().size())) {
            err_stream << "ERROR: Incorrect volume link to non-existent volume "
                       << sf.volume_link();
            throw std::invalid_argument(err_stream.str());
        }

        // A passive surface should have material, if the detector was
        // constructed with material
        if (!det.material_store().all_empty() &&
            (sf.is_passive() && !sf.has_material())) {
            std::cout << err_stream.str()
                      << "WARNING: Passive surface without material: "
                      << sf_descr << "\n"
                      << std::endl;
        }

        // Check that the same surface is registered in the detector surface
        // lookup
        const auto sf_from_lkp =
            tracking_surface{det, det.surface(sf.barcode())};
        if (sf_from_lkp != sf) {
            err_stream << "ERROR: Surfaces in volume and detector lookups "
                       << "differ:\n In volume acceleration data structure: "
                       << sf << "\nIn detector surface lookup: " << sf_from_lkp;

            throw std::runtime_error(err_stream.str());
        }
    }

    /// Test wether a given surface @param check_desc is properly registered at
    /// least once in one of the volume acceleration data structures
    ///
    /// @param ref_descr one of the surfaces in the volumes acceleration data
    /// @param check_descr surface that we are searching for
    /// @param success communicate success to the outside
    template <typename detector_t>
    DETRAY_HOST_DEVICE void operator()(
        const typename detector_t::surface_type &ref_descr,
        const typename detector_t::surface_type &check_descr, bool &success,
        const detector_t &det) const {

        // Check that the surface is being searched for in the right volume
        // The volume index of the ref_descr must be checked to be correct
        // beforehand, e.g. by the call operator above
        if (ref_descr.volume() != check_descr.volume()) {
            std::stringstream err_stream{};
            err_stream << "Incorrect volume index on surface: "
                       << tracking_surface{det, check_descr};

            throw std::invalid_argument(err_stream.str());
        }

        // Check if it is the surface we are looking for
        if (ref_descr == check_descr) {
            success = true;
        }
    }
};

/// A functor that checks the material parametrization for a surface/volume
struct material_checker {

    /// Error message for material consistency check
    template <typename material_t>
    void throw_material_error(const std::string &type, const dindex idx,
                              const material_t &mat) const {
        std::stringstream err_stream{};
        err_stream << "Invalid material found in: " << type << " at index "
                   << idx << ": " << mat;

        throw std::invalid_argument(err_stream.str());
    }

    /// Test wether a given material map contains invalid material
    ///
    /// @param material_coll collection of material grids
    /// @param idx the specific grid to be checked
    /// @param id type id of the material grid collection
    template <typename material_coll_t, typename index_t, typename id_t>
    requires concepts::material_map<typename material_coll_t::value_type>
        DETRAY_HOST_DEVICE void operator()(const material_coll_t &material_coll,
                                           const index_t idx,
                                           const id_t id) const {

        const auto mat_map = material_coll[idx];

        // Check wether there are any entries in the bins
        if (mat_map.size() == 0u) {
            std::stringstream err_stream{};
            err_stream << "Empty material grid: " << static_cast<int>(id)
                       << " at index " << idx;

            throw std::invalid_argument(err_stream.str());
        } else {
            for (const auto &bin : mat_map.bins()) {
                if (bin.size() == 0u) {
                    std::stringstream err_stream{};
                    err_stream << "Empty material bin: " << static_cast<int>(id)
                               << " at index " << idx;

                    throw std::invalid_argument(err_stream.str());
                }
            }
        }
    }

    /// Test wether a given collection of material contains invalid material
    ///
    /// @param material_coll collection of material slabs/rods/raw mat
    /// @param idx the specific instance to be checked
    template <typename material_coll_t, typename index_t, typename id_t>
    requires concepts::homogeneous_material<
        typename material_coll_t::value_type>
        DETRAY_HOST_DEVICE void operator()(const material_coll_t &material_coll,
                                           const index_t idx,
                                           const id_t) const {

        using material_t = typename material_coll_t::value_type;
        using scalar_t = typename material_t::scalar_type;

        const material_t &mat = material_coll.at(idx);

        // Homogeneous volume material
        if constexpr (concepts::material_params<material_t>) {

            if (mat == detray::vacuum<scalar_t>{}) {
                throw_material_error("homogeneous volume material", idx, mat);
            }

        } else {
            // Material slabs and rods
            if (!mat) {
                throw_material_error("homogeneous surface material", idx, mat);
            }
        }
    }
};

/// @brief Checks wether the data containers of a detector are empty
///
/// In case the default metadata is used, the unused containers are allowed to
/// be empty.
template <typename detector_t>
inline void check_empty(const detector_t &det, const bool verbose) {

    // Check if there is at least one portal in the detector
    auto find_portals = [&det]() {
        if (det.portals().empty()) {
            return false;
        }
        // In the brute force finder, also other surfaces can be contained, e.g.
        // passive surfaces (depends on the detector)
        return std::ranges::any_of(
            det.portals(), [](auto pt_desc) { return pt_desc.is_portal(); });
    };

    // Check if there is at least one volume in the detector volume finder
    auto find_volumes = [](const typename detector_t::volume_finder &vf) {
        return std::ranges::any_of(vf.all(), [](const auto &v) {
            return !detail::is_invalid_value(v);
        });
    };

    // Fatal errors
    if (det.volumes().empty()) {
        throw std::runtime_error("ERROR: No volumes in detector");
    }
    if (det.surfaces().empty()) {
        throw std::runtime_error("ERROR: No surfaces found");
    }
    if (det.transform_store().empty()) {
        throw std::runtime_error("ERROR: No transforms in detector");
    }
    if (det.mask_store().all_empty()) {
        throw std::runtime_error("ERROR: No masks in detector");
    }
    if (!find_portals()) {
        throw std::runtime_error("ERROR: No portals in detector");
    }

    // Warnings

    // Check the material description
    if (det.material_store().all_empty()) {
        std::cout << "WARNING: No material in detector\n" << std::endl;
    } else if (verbose) {
        // Check for empty material collections
        detail::report_empty(
            det.material_store(), "material store",
            std::make_index_sequence<detector_t::materials::n_types>{});
    }

    // Check for empty acceleration data structure collections (e.g. grids)
    if (verbose) {
        // Check for empty mask collections
        detail::report_empty(
            det.mask_store(), "mask store",
            std::make_index_sequence<detector_t::masks::n_types>{});

        detail::report_empty(
            det.accelerator_store(), "acceleration data structures store",
            std::make_index_sequence<detector_t::accel::n_types>{});
    }

    // Check volume search data structure
    if (!find_volumes(det.volume_search_grid())) {
        std::cout << "WARNING: No entries in volume finder\n" << std::endl;
    }
}

/// @brief Checks the internal consistency of a detector
template <typename detector_t>
inline bool check_consistency(const detector_t &det, const bool verbose = false,
                              const typename detector_t::name_map &names = {}) {
    check_empty(det, verbose);

    std::stringstream err_stream{};
    // Check the volumes
    for (const auto &[idx, vol_desc] :
         detray::views::enumerate(det.volumes())) {
        const auto vol = tracking_volume{det, vol_desc};
        err_stream << "VOLUME \"" << print_volume_name(vol, names) << "\":\n";

        // Check that nothing is obviously broken
        if (!vol.self_check(err_stream)) {
            throw std::invalid_argument(err_stream.str());
        }

        // Check consistency in the context of the owning detector
        if (vol.index() != idx) {
            err_stream << "ERROR: Incorrect volume index! Found volume:\n"
                       << vol << "\nat index " << idx;
            throw std::invalid_argument(err_stream.str());
        }

        // Go through the acceleration data structures and check the surfaces
        vol.template visit_surfaces<detail::surface_checker>(det, vol.index(),
                                                             names);

        // Check the volume material, if present
        if (vol.has_material()) {
            vol.template visit_material<detail::material_checker>(
                vol_desc.material().id());
        }
    }

    // Check the surfaces in the detector's surface lookup
    for (const auto &[idx, sf_desc] :
         detray::views::enumerate(det.surfaces())) {
        const auto sf = tracking_surface{det, sf_desc};
        const auto vol = tracking_volume{det, sf.volume()};
        err_stream << "VOLUME \"" << print_volume_name(vol, names) << "\":\n";

        // Check that nothing is obviously broken
        if (!sf.self_check(err_stream)) {
            err_stream << "\nat surface no. " << std::to_string(idx);
            throw std::invalid_argument(err_stream.str());
        }

        // Check consistency in the context of the owning detector
        if (sf.index() != idx) {
            err_stream << "ERROR: Incorrect surface index! Found surface:\n"
                       << sf << "\nat index " << idx;
            throw std::invalid_argument(err_stream.str());
        }

        // Check that the surface can be found in its volume's acceleration
        // data structures (if there are no grids, must at least be in the
        // brute force method)
        bool is_registered = false;

        vol.template visit_surfaces<detail::surface_checker>(
            sf_desc, is_registered, det);

        if (!is_registered) {
            err_stream << "ERROR: Found surface that is not part of its "
                       << "volume's navigation acceleration data structures:\n"
                       << "Surface: " << sf;
            throw std::invalid_argument(err_stream.str());
        }

        // Check the surface material, if present
        if (sf.has_material()) {
            sf.template visit_material<detail::material_checker>(
                sf_desc.material().id());
        }
    }

    return true;
}

}  // namespace detray::detail
