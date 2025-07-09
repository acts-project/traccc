/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// Detray include(s).
#include <any>
#include <detray/core/detector.hpp>
#include <detray/detectors/default_metadata.hpp>
#include <detray/detectors/telescope_metadata.hpp>
#include <detray/detectors/toy_metadata.hpp>

namespace traccc {

class move_only_any {
    public:
    move_only_any() = default;

    template <typename obj_t>
    explicit move_only_any(obj_t &&obj)
        : m_obj(std::malloc(sizeof(obj_t))),
          m_type(&typeid(obj_t)),
          m_destructor(get_destructor<obj_t>()) {
        new (m_obj) obj_t(std::move(obj));
    }

    ~move_only_any() {
        if (m_obj != nullptr) {
            assert(m_destructor != nullptr);
            m_destructor(m_obj);
            std::free(m_obj);
        }
    }

    template <typename obj_t>
    void set(obj_t &&obj) {
        if (m_obj != nullptr) {
            assert(m_destructor != nullptr);
            m_destructor(m_obj);
            std::free(m_obj);
        }

        m_obj = std::malloc(sizeof(obj_t));
        new (m_obj) obj_t(std::move(obj));

        m_type = &typeid(obj_t);
        m_destructor = get_destructor<obj_t>();
    }

    template <typename obj_t>
    bool is() const {
        return (*m_type == typeid(obj_t));
    }

    const std::type_info &type() const { return *m_type; }

    template <typename obj_t>
    obj_t &as() const {
        return *static_cast<obj_t *>(m_obj);
    }

    private:
    template <typename detector_t>
    void (*get_destructor() const)(void *) {
        return static_cast<void (*)(void *)>(
            [](void *ptr) { static_cast<detector_t *>(ptr)->~detector_t(); });
    }

    void *m_obj = nullptr;
    const std::type_info *m_type = nullptr;
    void (*m_destructor)(void *) = nullptr;
};

}  // namespace traccc
