/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/move_only_any.hpp"

// Detray include(s).
#include <any>
#include <detray/core/detector.hpp>
#include <detray/detectors/default_metadata.hpp>
#include <detray/detectors/telescope_metadata.hpp>
#include <detray/detectors/toy_metadata.hpp>

namespace traccc {

/// Typeless, owning, host detector object
class host_detector {
    public:
    host_detector() = default;

    template <typename detector_traits_t>
    void set(typename detector_traits_t::host&& obj) requires(
        is_detector_traits<detector_traits_t>) {
        m_obj.set<typename detector_traits_t::host>(std::move(obj));
    }

    template <typename detector_traits_t>
    bool is() const requires(is_detector_traits<detector_traits_t>) {
        return (type() == typeid(typename detector_traits_t::host));
    }

    const std::type_info& type() const { return m_obj.type(); }

    template <typename detector_traits_t>
    typename detector_traits_t::host& as() const
        requires(is_detector_traits<detector_traits_t>) {
        return m_obj.as<typename detector_traits_t::host>();
    }

    template <typename detector_traits_t>
    typename detector_traits_t::view as_view() const
        requires(is_detector_traits<detector_traits_t>) {
        return detray::get_data(as<detector_traits_t>());
    }

    private:
    move_only_any m_obj;
};
}  // namespace traccc
