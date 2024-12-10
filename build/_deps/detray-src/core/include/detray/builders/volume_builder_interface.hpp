/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/geometry/tracking_volume.hpp"

// System include(s)
#include <memory>

namespace detray {

template <typename detector_t>
class surface_factory_interface;

template <typename detector_t>
class volume_decorator;

/// @brief Interface for volume builders (and volume builder decorators)
template <typename detector_t>
class volume_builder_interface {

    // Access protected methods
    friend class volume_decorator<detector_t>;

    public:
    using scalar_type = typename detector_t::scalar_type;
    template <typename T, std::size_t N>
    using array_type = typename detector_t::template array_type<T, N>;

    virtual ~volume_builder_interface() = default;

    /// @returns the global index for the volume
    /// @note the correct index is only available after calling @c init_vol
    DETRAY_HOST
    virtual auto vol_index() const -> dindex = 0;

    /// Toggles whether sensitive surfaces are added to the brute force method
    DETRAY_HOST
    virtual void has_accel(bool toggle) = 0;

    /// @returns whether sensitive surfaces are added to the brute force method
    DETRAY_HOST
    virtual bool has_accel() const = 0;

    /// @returns reading access to the volume
    DETRAY_HOST
    virtual auto operator()() const -> const
        typename detector_t::volume_type & = 0;
    DETRAY_HOST
    virtual auto operator()() -> typename detector_t::volume_type & = 0;

    /// @brief Adds a volume and all of its contents to a detector
    DETRAY_HOST
    virtual auto build(detector_t &det,
                       typename detector_t::geometry_context ctx = {}) ->
        typename detector_t::volume_type * = 0;

    /// @brief Add the transform for the volume placement - copy
    DETRAY_HOST
    virtual void add_volume_placement(
        const typename detector_t::transform3_type &trf = {}) = 0;

    /// @brief Add the transform for the volume placement from the translation
    /// @param t. The rotation will be the identity matrix.
    DETRAY_HOST
    virtual void add_volume_placement(
        const typename detector_t::point3_type &t) = 0;

    /// @brief Add the transform for the volume placement from the translation
    /// @param t , the new x- and z-axis (@param x, @param z).
    DETRAY_HOST
    virtual void add_volume_placement(
        const typename detector_t::point3_type &t,
        const typename detector_t::vector3_type &x,
        const typename detector_t::vector3_type &z) = 0;

    /// @brief Add surfaces to the volume
    /// @returns the index range of the sensitives in the temporary surface
    /// container used by the factory ( gets final update in @c build() )
    DETRAY_HOST
    virtual void add_surfaces(
        std::shared_ptr<surface_factory_interface<detector_t>> sf_factory,
        typename detector_t::geometry_context ctx = {}) = 0;

    protected:
    /// Access to builder data
    /// @{
    virtual typename detector_t::surface_lookup_container &surfaces() = 0;
    virtual typename detector_t::transform_container &transforms() = 0;
    virtual typename detector_t::mask_container &masks() = 0;
    /// @}
};

/// @brief Decorator for the volume builder.
///
/// Can be volume builders that introduce special sorting/memory layout, or
/// accelerator builders, like the grid builder.
template <typename detector_t>
class volume_decorator : public volume_builder_interface<detector_t> {

    public:
    using scalar_type = typename detector_t::scalar_type;
    template <typename T, std::size_t N>
    using array_type = typename detector_t::template array_type<T, N>;

    DETRAY_HOST
    explicit volume_decorator(
        std::unique_ptr<volume_builder_interface<detector_t>> vol_builder)
        : m_builder(std::move(vol_builder)) {
        assert(m_builder != nullptr);
    }

    DETRAY_HOST
    auto operator()() -> typename detector_t::volume_type & override {
        return m_builder->operator()();
    }

    DETRAY_HOST
    auto operator()() const -> const
        typename detector_t::volume_type & override {
        return m_builder->operator()();
    }

    DETRAY_HOST
    auto vol_index() const -> dindex override { return m_builder->vol_index(); }

    DETRAY_HOST
    void has_accel(bool toggle) override { m_builder->has_accel(toggle); };

    DETRAY_HOST
    bool has_accel() const override { return m_builder->has_accel(); }

    DETRAY_HOST
    auto build(detector_t &det,
               typename detector_t::geometry_context /*ctx*/ = {}) ->
        typename detector_t::volume_type * override {
        return m_builder->build(det);
    }

    DETRAY_HOST
    void add_volume_placement(
        const typename detector_t::transform3_type &trf = {}) override {
        return m_builder->add_volume_placement(trf);
    }

    DETRAY_HOST
    void add_volume_placement(
        const typename detector_t::point3_type &t) override {
        return m_builder->add_volume_placement(t);
    }

    DETRAY_HOST
    void add_volume_placement(
        const typename detector_t::point3_type &t,
        const typename detector_t::vector3_type &x,
        const typename detector_t::vector3_type &z) override {
        return m_builder->add_volume_placement(t, x, z);
    }

    DETRAY_HOST
    void add_surfaces(
        std::shared_ptr<surface_factory_interface<detector_t>> sf_factory,
        typename detector_t::geometry_context ctx = {}) override {
        return m_builder->add_surfaces(std::move(sf_factory), ctx);
    }
    /// @}

    protected:
    /// Access to underlying builder
    /// @{
    volume_builder_interface<detector_t> *get_builder() {
        return m_builder.get();
    }
    typename detector_t::surface_lookup_container &surfaces() override {
        return m_builder->surfaces();
    }
    typename detector_t::transform_container &transforms() override {
        return m_builder->transforms();
    }
    typename detector_t::mask_container &masks() override {
        return m_builder->masks();
    }
    /// @}

    private:
    std::unique_ptr<volume_builder_interface<detector_t>> m_builder;
};

}  // namespace detray
