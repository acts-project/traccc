/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s)
#include "detray/core/detector.hpp"
#include "detray/navigation/volume_graph.hpp"

// Detray IO include(s)
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/io/utils/create_path.hpp"

// Detray plugin include(s)
#include "detray/plugins/svgtools/illustrator.hpp"
#include "detray/plugins/svgtools/writer.hpp"

// Detray test include(s)
#include "detray/options/detector_io_options.hpp"
#include "detray/options/parse_options.hpp"

// Vecmem include(s)
#include <vecmem/memory/host_memory_resource.hpp>

// Actsvg include(s)
#include <actsvg/core.hpp>

// Boost
#include "detray/options/boost_program_options.hpp"

// System include(s)
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>

namespace po = boost::program_options;

using namespace detray;

int main(int argc, char** argv) {

    // Use the most general type to be able to read in all detector files
    using detector_t = detray::detector<>;

    // Visualization style to be applied to the svgs
    auto style = detray::svgtools::styling::tableau_colorblind::style;

    // Specific options for this test
    po::options_description desc("\ndetray detector validation options");

    std::vector<dindex> volumes;
    std::vector<dindex> surfaces;
    std::vector<dindex> window;
    desc.add_options()("outdir", po::value<std::string>(),
                       "Output directory for plots")(
        "context", po::value<dindex>(), "Number of the geometry context")(
        "search_window", po::value<std::vector<dindex>>(&window)->multitoken(),
        "Size of the grid surface search window")(
        "volumes", po::value<std::vector<dindex>>(&volumes)->multitoken(),
        "List of volumes that should be displayed")(
        "surfaces", po::value<std::vector<dindex>>(&surfaces)->multitoken(),
        "List of surfaces that should be displayed")(
        "hide_portals", "Hide portal surfaces")("hide_passives",
                                                "Hide passive surfaces")(
        "hide_material", "Don't draw surface material")(
        "hide_eta_lines", "Hide eta lines")("show_info", "Show info boxes")(
        "write_volume_graph", "Writes the volume graph to file");

    // Configs to be filled
    detray::io::detector_reader_config reader_cfg{};
    // Also display incorrect geometries for debugging
    reader_cfg.do_check(false);

    po::variables_map vm =
        detray::options::parse_options(desc, argc, argv, reader_cfg);

    // General options
    std::string outdir{vm.count("outdir") ? vm["outdir"].as<std::string>()
                                          : "./plots/"};
    auto path = detray::io::create_path(outdir);

    // The geometry context to be displayed
    detector_t::geometry_context gctx;
    if (vm.count("context")) {
        gctx = detector_t::geometry_context{vm["context"].as<dindex>()};
    }
    // Grid neighborhood size
    if (vm.count("search_window")) {
        if (window.size() != 2u) {
            throw std::invalid_argument(
                "Incorrect surface grid search window. Please provide two "
                "integer distances.");
        }
    } else {
        // default
        window = {1u, 1u};
    }

    // Read the detector geometry
    vecmem::host_memory_resource host_mr;

    const auto [det, names] =
        detray::io::read_detector<detector_t>(host_mr, reader_cfg);

    // Creating the svg generator for the detector.
    detray::svgtools::illustrator il{det, names, style};
    il.show_info(vm.count("show_info"));
    il.hide_eta_lines(vm.count("hide_eta_lines"));
    il.hide_portals(vm.count("hide_portals"));
    il.hide_passives(vm.count("hide_passives"));
    il.hide_material(!vm.count("material_file") || vm.count("hide_material"));
    il.hide_grids(!vm.count("grid_file"));
    il.search_window({window[0], window[1]});

    actsvg::style::stroke stroke_black = actsvg::style::stroke();
    actsvg::style::font axis_font = actsvg::style::font();
    axis_font._size = 35u;

    // x-y axis.
    auto xy_axis = actsvg::draw::x_y_axes("axes", {-1100, 1100}, {-1100, 1100},
                                          stroke_black, "x", "y", axis_font);
    // z-r axis.
    auto zr_axis = actsvg::draw::x_y_axes("axes", {-3100, 3100}, {-5, 1100},
                                          stroke_black, "z", "r", axis_font);
    // z-phi axis.
    auto zphi_axis =
        actsvg::draw::x_y_axes("axes", {-3100, 3100}, {-1000, 1000},
                               stroke_black, "z", "phi", axis_font);

    // Creating the views.
    const actsvg::views::x_y xy;
    const actsvg::views::z_r zr;
    const actsvg::views::z_phi zphi;

    // Display the volumes
    if (!volumes.empty()) {
        const auto [vol_xy_svg, xy_sheets] = il.draw_volumes(volumes, xy, gctx);
        detray::svgtools::write_svg(path / vol_xy_svg._id,
                                    {xy_axis, vol_xy_svg});
        for (const auto& sheet : xy_sheets) {
            detray::svgtools::write_svg(path / sheet._id, sheet);
        }

        const auto [vol_zr_svg, _sh] = il.draw_volumes(volumes, zr, gctx);
        detray::svgtools::write_svg(path / vol_zr_svg._id,
                                    {zr_axis, vol_zr_svg});

        const auto [_vol, zphi_sheets] = il.draw_volumes(volumes, zphi, gctx);
        for (const auto& sheet : zphi_sheets) {
            detray::svgtools::write_svg(path / sheet._id, sheet);
        }
    }

    // Display the surfaces
    if (!surfaces.empty()) {
        const auto [sf_xy_svg, mat_xy_svg] =
            il.draw_surfaces(surfaces, xy, gctx);
        detray::svgtools::write_svg(path / sf_xy_svg._id, {xy_axis, sf_xy_svg});
        detray::svgtools::write_svg(path / mat_xy_svg._id,
                                    {xy_axis, mat_xy_svg});

        [[maybe_unused]] const auto [sf_zr_svg, mat_zr_svg] =
            il.draw_surfaces(surfaces, zr, gctx);
        detray::svgtools::write_svg(path / sf_zr_svg._id, {zr_axis, sf_zr_svg});

        [[maybe_unused]] const auto [sf_zphi_svg, mat_zphi_svg] =
            il.draw_surfaces(surfaces, zphi, gctx);
        detray::svgtools::write_svg(path / mat_zphi_svg._id,
                                    {zphi_axis, mat_zphi_svg});
    }

    // If nothing was specified, display the whole detector
    if (volumes.empty() && surfaces.empty()) {
        const auto det_xy_svg = il.draw_detector(xy, gctx);
        detray::svgtools::write_svg(path / det_xy_svg._id,
                                    {xy_axis, det_xy_svg});

        const auto det_zr_svg = il.draw_detector(zr, gctx);
        detray::svgtools::write_svg(path / det_zr_svg._id,
                                    {zr_axis, det_zr_svg});
    }

    // Display the detector volume graph
    if (vm.count("write_volume_graph")) {
        detray::volume_graph graph(det);

        detray::io::file_handle stream{
            path / (det.name(names) + "_volume_graph.dot"),
            std::ios::out | std::ios::trunc};
        *stream << graph.to_dot_string();
    }
}
