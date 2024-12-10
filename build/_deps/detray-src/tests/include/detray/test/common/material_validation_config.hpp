/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Detray test include(s).
#include "detray/test/common/detail/whiteboard.hpp"
#include "detray/test/common/fixture_base.hpp"
#include "detray/test/utils/types.hpp"

// Vecmem include(s)
#include <vecmem/memory/memory_resource.hpp>

// System include(s)
#include <limits>
#include <memory>
#include <string>

namespace detray::test {

/// @brief Configuration for a detector scan test.
struct material_validation_config : public test::fixture_base<>::configuration {
    using base_type = test::fixture_base<>;
    using scalar_type = typename base_type::scalar;
    using vector3_type = typename base_type::vector3;

    /// Name of the test
    std::string m_name{"material_validation"};
    /// Vecmem memory resource for the device allocations
    vecmem::memory_resource *m_dev_mr{nullptr};
    /// Access to truth data and tracks
    std::shared_ptr<test::whiteboard> m_white_board;
    /// Name of the output file, containing the complete ray material traces
    std::string m_material_file{"navigation_material_trace.csv"};
    /// The maximal number of test tracks to run
    std::size_t m_n_tracks{detray::detail::invalid_value<std::size_t>()};
    /// Allowed relative discrepancy between truth and navigation material
    scalar_type m_rel_error{0.001f};

    /// Getters
    /// @{
    const std::string &name() const { return m_name; }
    vecmem::memory_resource *device_mr() const { return m_dev_mr; }
    std::shared_ptr<test::whiteboard> whiteboard() { return m_white_board; }
    std::shared_ptr<test::whiteboard> whiteboard() const {
        return m_white_board;
    }
    const std::string &material_file() const { return m_material_file; }
    std::size_t n_tracks() const { return m_n_tracks; }
    scalar_type relative_error() const { return m_rel_error; }
    /// @}

    /// Setters
    /// @{
    material_validation_config &name(const std::string &n) {
        m_name = n;
        return *this;
    }
    material_validation_config &device_mr(vecmem::memory_resource *mr) {
        m_dev_mr = mr;
        return *this;
    }
    material_validation_config &whiteboard(
        std::shared_ptr<test::whiteboard> w_board) {
        if (!w_board) {
            throw std::invalid_argument(
                "Material validation: No valid whiteboard instance");
        }
        m_white_board = std::move(w_board);
        return *this;
    }
    material_validation_config &material_file(const std::string &f) {
        m_material_file = f;
        return *this;
    }
    material_validation_config &n_tracks(std::size_t n) {
        m_n_tracks = n;
        return *this;
    }
    material_validation_config &relative_error(scalar_type re) {
        assert(re > 0.f);
        m_rel_error = re;
        return *this;
    }
    /// @}
};

}  // namespace detray::test
