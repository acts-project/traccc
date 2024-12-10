/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "detray/geometry/tracking_surface.hpp"
#include "detray/plugins/svgtools/conversion/detector.hpp"
#include "detray/plugins/svgtools/conversion/grid.hpp"
#include "detray/plugins/svgtools/conversion/information_section.hpp"
#include "detray/plugins/svgtools/conversion/intersection.hpp"
#include "detray/plugins/svgtools/conversion/landmark.hpp"
#include "detray/plugins/svgtools/conversion/surface.hpp"
#include "detray/plugins/svgtools/conversion/surface_material.hpp"
#include "detray/plugins/svgtools/conversion/trajectory.hpp"
#include "detray/plugins/svgtools/conversion/volume.hpp"
#include "detray/plugins/svgtools/meta/display/geometry.hpp"
#include "detray/plugins/svgtools/meta/display/information.hpp"
#include "detray/plugins/svgtools/meta/display/tracking.hpp"
#include "detray/plugins/svgtools/meta/proto/eta_lines.hpp"
#include "detray/plugins/svgtools/styling/styling.hpp"
#include "detray/plugins/svgtools/utils/groups.hpp"
#include "detray/utils/ranges.hpp"

// Actsvg include(s)
#include "actsvg/meta.hpp"

// System include(s)
#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

namespace detray::svgtools {

/// @brief SVG generator for a detector and related entities.
///
/// Provides an easy interface for displaying typical objects in a detector.
/// For more flexibility, use the tools in svgtools::conversion to convert
/// detray objects to their respective proto object.
/// Use functions in svgtools::meta::display and actsvg::display to display
/// the proto objects.
///
/// @note Avoid using ids containing spaces or dashes as this seems to cause
/// issues (for instance regarding information boxes). Furthermore, to view
/// information boxes, they must be enabled in the constructor. Furthermore the
/// svg viewer (opening the file after it is created) must support animations.
template <typename detector_t>
class illustrator {

    using point3 = typename detector_t::point3_type;
    using point3_container = std::vector<point3>;

    public:
    illustrator() = delete;

    /// @param detector the detector
    /// @param name_map naming scheme of the detector
    /// @param style display style
    ///
    /// @note information boxes are enabled by default
    illustrator(const detector_t& detector,
                const typename detector_t::name_map& name_map,
                const styling::style& style =
                    detray::svgtools::styling::tableau_colorblind::style)
        : _detector{detector}, _name_map{name_map}, _style{style} {}

    /// @returns the detector and volume names
    const typename detector_t::name_map& names() { return _name_map; }

    /// Access the illustrator style
    styling::style& style() { return _style; }

    /// Toggle info boxes
    void show_info(bool toggle = true) { _show_info = toggle; }
    /// Toggle eta lines in detector rz-view
    void hide_eta_lines(bool toggle = true) { _hide_eta_lines = toggle; }
    /// Toggle surface grids
    void hide_grids(bool toggle = true) { _hide_grids = toggle; }
    /// Toggle surface material
    void hide_material(bool toggle = true) { _hide_material = toggle; }
    /// Toggle portal surfaces
    void hide_portals(bool toggle = true) { _hide_portals = toggle; }
    /// Toggle passive surfaces
    void hide_passives(bool toggle = true) { _hide_passives = toggle; }
    /// Neighborhood search window for the grid display
    void search_window(std::array<dindex, 2> window) {
        _search_window = window;
    }
    /// @returns the detector name
    const std::string& det_name() const { return _detector.name(_name_map); }

    /// @brief Converts a single detray surface of the detector to an svg.
    ///
    /// @param index the index of the surface in the detector.
    /// @param view the display view.
    /// @param gctx the geometry context.
    ///
    /// @returns @c actsvg::svg::object of the detector's surface.
    template <typename view_t>
    inline auto draw_surface(
        const dindex index, const view_t& view,
        const typename detector_t::geometry_context& gctx = {}) const {

        const auto surface =
            detray::tracking_surface<detector_t>{_detector, index};

        actsvg::svg::object ret;
        actsvg::svg::object material;
        const auto& style = _style._detector_style._volume_style;

        if (surface.is_portal()) {
            auto p_portal = svgtools::conversion::portal(
                gctx, _detector, surface, view, style._portal_style, false,
                _hide_material);

            // Draw the portal directly
            std::string id = p_portal._name + "_" + svg_id(view);
            ret = actsvg::display::portal(std::move(id), p_portal, view);

            if (!_hide_material) {
                material = actsvg::display::surface_material(
                    id + "_material_map", p_portal._surface._material);
            }
        } else {
            const auto& sf_style = surface.is_sensitive()
                                       ? style._sensitive_surface_style
                                       : style._passive_surface_style;

            auto p_surface = svgtools::conversion::surface(
                gctx, _detector, surface, view, sf_style, _hide_material);

            // Draw the surface directly
            std::string id = p_surface._name + "_" + svg_id(view);
            ret = actsvg::display::surface(std::move(id), p_surface, view);

            if (!_hide_material) {
                material = actsvg::display::surface_material(
                    id + "_material_map", p_surface._material);
            }
        }
        // Add an optional info box
        if (_show_info) {
            auto p_information_section =
                svgtools::conversion::information_section(gctx, surface);

            auto info_box = svgtools::meta::display::information_section(
                ret._id + "_info_box", p_information_section, view,
                _info_screen_offset, ret);
            ret.add_object(info_box);
        }

        return std::tuple{ret, material};
    }

    /// @brief Converts a multiple of detray surfaces of the detector to an svg.
    ///
    /// @param indices the collection of surface indices in the detector to
    /// convert.
    /// @param view the display view.
    /// @param gctx the geometry context.
    ///
    /// @returns @c actsvg::svg::object of the detector's surfaces.
    template <detray::ranges::range range_t, typename view_t>
    inline auto draw_surfaces(
        const range_t& indices, const view_t& view,
        const typename detector_t::geometry_context& gctx = {}) const {

        auto ret =
            svgtools::utils::group(det_name() + "_surfaces_" + svg_id(view));

        auto material =
            svgtools::utils::group(det_name() + "_material_" + svg_id(view));

        for (const auto [i, index] : detray::views::enumerate(indices)) {
            auto [sf_svg, mat_svg] = draw_surface(index, view, gctx);

            ret.add_object(sf_svg);

            // Material is optional
            if (mat_svg.is_defined()) {
                // Only add one gradient box
                if (i == 0) {
                    material.add_object(mat_svg);
                } else {
                    material.add_object(mat_svg._sub_objects[0]);
                }
            }
        }

        return std::tuple{ret, material};
    }

    /// @brief Converts the material map of a single detray surface to an svg.
    ///
    /// @param index the index of the surface in the detector.
    /// @param view the display view.
    /// @param gctx the geometry context.
    ///
    /// @returns @c actsvg::svg::object of the surface's material map.
    template <typename view_t>
    inline auto draw_surface_material(const dindex index,
                                      const view_t& view) const {

        const auto surface =
            detray::tracking_surface<detector_t>{_detector, index};

        if (_hide_material) {
            return actsvg::svg::object{};
        }

        const styling::surface_material_style* mat_style{nullptr};
        const auto& vol_style = _style._detector_style._volume_style;
        switch (surface.id()) {
            using enum surface_id;
            case e_portal: {
                mat_style =
                    &vol_style._portal_style._surface_style._material_style;
                break;
            }
            case e_sensitive: {
                mat_style = &vol_style._sensitive_surface_style._material_style;
                break;
            }
            case e_passive: {
                mat_style = &vol_style._passive_surface_style._material_style;
                break;
            }
            case e_unknown: {
                throw std::runtime_error(
                    "Encountered surface of unknown type.");
                break;
            }
        }

        auto p_material = svgtools::conversion::surface_material(
            _detector, surface, view, *mat_style);

        std::string id = det_name() + "_material_map_" +
                         std::to_string(surface.index()) + svg_id(view);

        return actsvg::display::surface_material(std::move(id), p_material);
    }

    /// @brief Converts the material of multiple detray surfaces to an svg.
    ///
    /// @param indices the collection of surface indices in the detector to
    /// convert.
    /// @param view the display view.
    ///
    /// @returns @c actsvg::svg::object of the surface's material maps.
    template <detray::ranges::range range_t, typename view_t>
    inline auto draw_surface_materials(const range_t& indices,
                                       const view_t& view) const {

        auto ret = svgtools::utils::group(det_name() + "_surface_materials_" +
                                          svg_id(view));

        for (const auto [i, index] : detray::views::enumerate(indices)) {
            auto mat_svg = draw_surface_material(index, view);

            // Only add one gradient box
            if (i == 0) {
                ret.add_object(mat_svg);
            } else {
                ret.add_object(mat_svg._sub_objects[0]);
            }
        }

        return ret;
    }

    /// @brief Converts a detray volume of the detector to an svg.
    ///
    /// @param index the index of the volume in the detector.
    /// @param view the display view.
    /// @param gctx the geometry context.
    ///
    /// @returns @c actsvg::svg::object of the detector's volume.
    template <typename view_t>
    inline auto draw_volume(
        const dindex index, const view_t& view,
        const typename detector_t::geometry_context& gctx = {}) const {

        const auto d_volume = tracking_volume{_detector, index};

        auto [p_volume, gr_type] = svgtools::conversion::volume(
            gctx, _detector, d_volume, view,
            _style._detector_style._volume_style, _hide_portals, _hide_passives,
            _hide_grids, _hide_material, _search_window);

        // Draw the basic volume
        p_volume._name = d_volume.name(_name_map);
        std::string id = p_volume._name + "_" + svg_id(view);
        auto vol_svg = svgtools::utils::group(id);

        [[maybe_unused]] auto display_mode =
            _hide_grids ? actsvg::display::e_module_info
                        : actsvg::display::e_grid_info;

        actsvg::svg::object sheet;

        // zr and xy - views of volume including the portals
        if constexpr (!std::is_same_v<view_t, actsvg::views::z_phi>) {

            vol_svg.add_object(actsvg::display::volume(id, p_volume, view));

            if (!_hide_grids) {
                auto grid_svg =
                    actsvg::display::grid(id + "_grid", p_volume._surface_grid);
                vol_svg.add_object(grid_svg);
            }

            // Display the surfaces connected to the grid: in zphi-view for
            // barrel, and xy-view for endcaps
        } else if (gr_type == conversion::detail::grid_type::e_barrel) {
            p_volume._name = det_name() + "_" + p_volume._name;
            sheet = actsvg::display::barrel_sheet(
                p_volume._name + "_sheet_zphi", p_volume, {800, 800},
                display_mode);
        }
        if constexpr (std::is_same_v<view_t, actsvg::views::x_y>) {
            if (gr_type == conversion::detail::grid_type::e_endcap) {
                p_volume._name = det_name() + "_" + p_volume._name;
                sheet = actsvg::display::endcap_sheet(
                    p_volume._name + "_sheet_xy", p_volume, {800, 800},
                    display_mode);
            }
        }

        return std::tuple{vol_svg, sheet};
    }

    /// @brief Converts multiple detray volumes of the detector to an svg.
    ///
    /// @param indices the collection of volume indices in the detector to
    /// convert.
    /// @param view the display view.
    /// @param gctx the geometry context.
    ///
    /// @returns @c actsvg::svg::object of the detector's volumes.
    template <detray::ranges::range range_t, typename view_t>
    inline auto draw_volumes(
        const range_t& indices, const view_t& view,
        const typename detector_t::geometry_context& gctx = {}) const {

        // Overlay the volume svgs
        auto vol_group =
            svgtools::utils::group(det_name() + "_volumes_" + svg_id(view));
        // The surface[grid] sheets per volume
        std::vector<actsvg::svg::object> sheets;

        for (const auto index : indices) {

            auto [vol_svg, sheet] = draw_volume(index, view, gctx);

            // The general volume display
            if constexpr (!std::is_same_v<view_t, actsvg::views::z_phi>) {
                vol_group.add_object(vol_svg);
            }
            sheets.push_back(std::move(sheet));
        }

        return std::tuple(vol_group, sheets);
    }

    /// @brief Converts a detray detector to an svg.
    ///
    /// @param view the display view.
    /// @param gctx the geometry context.
    ///
    /// @returns @c actsvg::svg::object of the detector.
    template <typename view_t>
    inline auto draw_detector(
        const view_t& view,
        const typename detector_t::geometry_context& gctx = {}) const {

        auto p_detector = svgtools::conversion::detector(
            gctx, _detector, view, _style._detector_style, _hide_portals,
            _hide_passives, _hide_grids);

        std::string id = det_name() + "_" + svg_id(view);

        auto det_svg =
            actsvg::display::detector(std::move(id), p_detector, view);

        if constexpr (std::is_same_v<view_t, actsvg::views::z_r>) {
            if (!_hide_eta_lines) {

                auto p_eta_lines = svgtools::meta::proto::eta_lines{};

                svgtools::styling::apply_style(p_eta_lines,
                                               _style._eta_lines_style);

                // Hardcoded until we find a way to scale axes automatically
                p_eta_lines._r = 1100.f;
                p_eta_lines._z = 3100.f;

                det_svg.add_object(svgtools::meta::display::eta_lines(
                    "eta_lines_", p_eta_lines));
            }
        }

        return det_svg;
    }

    /// @brief Converts a point to an svg.
    ///
    /// @param prefix the id of the svg object.
    /// @param point the point.
    /// @param view the display view.
    ///
    /// @return actsvg::svg::object of the point.
    template <typename view_t, typename point_t>
    inline auto draw_landmark(const std::string& prefix, const point_t& point,
                              const view_t& view) const {
        auto p_landmark =
            svgtools::conversion::landmark(point, _style._landmark_style);

        return svgtools::meta::display::landmark(prefix + "_landmark",
                                                 p_landmark, view);
    }

    /// @brief Converts a collection of intersections to an svg.
    ///
    /// @param prefix the id of the svg object.
    /// @param intersections the intersection collection.
    /// @param dir the direction of the trajectory.
    /// @param view the display view.
    /// @param gctx the geometry context.
    ///
    /// @return @c actsvg::svg::object of the intersectio record.
    template <typename view_t, typename intersection_t>
    inline auto draw_intersections(
        const std::string& prefix,
        const std::vector<intersection_t>& intersections,
        const typename detector_t::vector3_type dir, const view_t& view,
        const typename detector_t::geometry_context& gctx = {}) const {

        auto p_ir = svgtools::conversion::intersection(
            _detector, intersections, dir, gctx, _style._intersection_style);
        // The first intersection sits in the origin by convention
        p_ir._landmarks.front()._position = {0.f, 0.f, 0.f};

        return svgtools::meta::display::intersection(prefix, p_ir, view);
    }

    /// @brief Converts a trajectory to an svg.
    ///
    /// @param prefix the id of the svg object.
    /// @param trajectory the trajectory (eg. ray or helix).
    /// @param path maximal pathlength of along the trajectory.
    /// @param view the display view.
    ///
    /// @return @c actsvg::svg::object of the trajectory.
    template <typename view_t, template <typename> class trajectory_t,
              typename algebra_t>
    inline auto draw_trajectory(const std::string& prefix,
                                const trajectory_t<algebra_t>& trajectory,
                                const dscalar<algebra_t> path,
                                const view_t& view) const {
        auto p_trajectory = svgtools::conversion::trajectory(
            trajectory, _style._trajectory_style, path);

        return svgtools::meta::display::trajectory(prefix + "_traj",
                                                   p_trajectory, view);
    }

    /// @brief Converts a trajectory to an svg.
    ///
    /// @param prefix the id of the svg object.
    /// @param trajectory the trajectory (eg. ray or helix).
    /// @param view the display view.
    ///
    /// @return actsvg::svg::object of the trajectory.
    template <typename view_t, typename point3_container>
    inline auto draw_trajectory(const std::string& prefix,
                                const point3_container& points,
                                const view_t& view) const {

        auto p_trajectory =
            svgtools::conversion::trajectory(points, _style._trajectory_style);

        return svgtools::meta::display::trajectory(prefix + "_traj",
                                                   p_trajectory, view);
    }

    /// @brief Converts a trajectory and its intersections to an svg with a
    /// related coloring.
    ///
    /// @param prefix the id of the svg object.
    /// @param intersections the intersection record.
    /// @param trajectory the trajectory (eg. ray or helix).
    /// @param view the display view.
    /// @param max_path the maximal path length from trajectory origin.
    /// @param gctx the geometry context.
    ///
    /// @return @c actsvg::svg::object of the trajectory and intersections
    template <typename view_t, class trajectory_t, typename intersection_t>
    inline auto draw_intersections_and_trajectory(
        const std::string& prefix,
        const std::vector<intersection_t>& intersections,
        const trajectory_t& trajectory, const view_t& view,
        typename detector_t::scalar_type max_path = 500.f,
        const typename detector_t::geometry_context& gctx = {}) const {

        actsvg::svg::object ret;
        ret._tag = "g";
        ret._id = prefix;

        if (!intersections.empty()) {

            // Draw intersections with landmark style
            auto p_ir = svgtools::conversion::intersection(
                _detector, intersections, trajectory.dir(0.f), gctx,
                _style._landmark_style);
            // The first intersection sits in the origin by convention
            p_ir._landmarks.front()._position = {0.f, 0.f, 0.f};

            ret.add_object(svgtools::meta::display::intersection(
                prefix + "_record", p_ir, view));

            // The intersection record is always sorted by path length
            const auto sf{detray::tracking_surface{
                _detector, intersections.back().sf_desc}};
            const auto final_pos = sf.local_to_global(
                gctx, intersections.back().local, trajectory.dir(0.f));
            max_path = getter::norm(final_pos - trajectory.pos(0.f));
        }

        ret.add_object(draw_trajectory(prefix + "_trajectory", trajectory,
                                       max_path, view));
        return ret;
    }

    private:
    /// @returns the string id of a view
    template <typename view_t>
    std::string svg_id(const view_t& view) const {
        return view._axis_names[0] + view._axis_names[1];
    }

    const actsvg::point2 _info_screen_offset{-300, 300};
    const detector_t& _detector;
    const typename detector_t::name_map& _name_map;
    bool _show_info = true;
    bool _hide_eta_lines = false;
    bool _hide_grids = false;
    bool _hide_material = true;
    bool _hide_portals = false;
    bool _hide_passives = false;
    std::array<dindex, 2> _search_window = {2u, 2u};
    styling::style _style = styling::tableau_colorblind::style;
};

}  // namespace detray::svgtools
