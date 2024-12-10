/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/builders/grid_factory.hpp"
#include "detray/builders/volume_builder.hpp"
#include "detray/builders/volume_builder_interface.hpp"
#include "detray/core/detector.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/definitions/geometry.hpp"
#include "detray/utils/grid/detail/concepts.hpp"
#include "detray/utils/type_traits.hpp"

// Vecmem include(s)
#include <vecmem/memory/memory_resource.hpp>

// System include(s)
#include <memory>
#include <vector>

namespace detray {

/// @brief Provides functionality to build a detray detector volume by volume
///
/// @tparam metadata the type definitions for the detector
/// @tparam bfield_bknd_t the type of magnetic field to be used
/// @tparam volume_builder_t the basic volume builder to be used for the
///                          geometry data
/// @tparam volume_data_t the data structure that holds the volume builders
template <typename metadata = default_metadata,
          template <typename> class volume_builder_t = volume_builder,
          template <typename...> class volume_data_t = std::vector>
class detector_builder {
    public:
    using detector_type = detector<metadata, host_container_types>;

    /// Add a new volume builder that will build a volume of the shape given by
    /// @param id
    template <typename... Args>
    DETRAY_HOST auto new_volume(const volume_id id, Args&&... args)
        -> volume_builder_interface<detector_type>* {

        m_volumes.push_back(std::make_unique<volume_builder_t<detector_type>>(
            id, static_cast<dindex>(m_volumes.size()),
            std::forward<Args>(args)...));

        return m_volumes.back().get();
    }

    /// @returns the number of volumes currently registered in the builder
    DETRAY_HOST auto n_volumes() const -> dindex {
        return static_cast<dindex>(m_volumes.size());
    }

    /// @returns 'true' if there is a volume builder registered for
    /// the volume with index @param volume_idx
    DETRAY_HOST bool has_volume(const std::size_t volume_idx) const {
        return volume_idx < m_volumes.size();
    }

    /// Decorate a volume builder at position @param volume_idx with more
    /// functionality
    template <class builder_t>
    DETRAY_HOST auto decorate(dindex volume_idx) -> builder_t* {

        m_volumes[volume_idx] =
            std::make_unique<builder_t>(std::move(m_volumes[volume_idx]));

        // Always works, we set it as this type in the line above
        return dynamic_cast<builder_t*>(m_volumes[volume_idx].get());
    }

    /// Decorate a volume builder @param v_builder with more functionality
    template <class builder_t>
    DETRAY_HOST auto decorate(
        const volume_builder_interface<detector_type>* v_builder)
        -> builder_t* {
        assert(v_builder != nullptr);

        return decorate<builder_t>(v_builder->vol_index());
    }

    /// Access a particular volume builder by volume index @param volume_idx
    DETRAY_HOST
    auto operator[](dindex volume_idx)
        -> volume_builder_interface<detector_type>* {
        return m_volumes[volume_idx].get();
    }

    /// Assembles the final detector from the volumes builders and allocates
    /// the detector containers with the memory resource @param resource
    DETRAY_HOST
    auto build(vecmem::memory_resource& resource) -> detector_type {

        detector_type det{resource};

        for (auto& vol_builder : m_volumes) {
            vol_builder->build(det);
        }

        det.set_volume_finder(std::move(m_vol_finder));

        // TODO: Add sorting, data deduplication etc. here later...

        return det;
    }

    /// Put the volumes into a search data structure
    template <typename... Args>
    DETRAY_HOST void set_volume_finder([[maybe_unused]] Args&&... args) {

        using vol_finder_t = typename detector_type::volume_finder;

        // Add dummy volume grid for now
        if constexpr (concepts::grid<vol_finder_t>) {

            // TODO: Construct it correctly with the grid builder
            mask<cylinder3D> vgrid_dims{0u,      0.f,   -constant<scalar>::pi,
                                        -2000.f, 180.f, constant<scalar>::pi,
                                        2000.f};
            std::array<std::size_t, 3> n_vgrid_bins{1u, 1u, 1u};

            std::array<std::vector<scalar>, 3UL> bin_edges{
                std::vector<scalar>{0.f, 180.f},
                std::vector<scalar>{-constant<scalar>::pi,
                                    constant<scalar>::pi},
                std::vector<scalar>{-2000.f, 2000.f}};

            grid_factory_type<vol_finder_t> vgrid_factory{};
            m_vol_finder = vgrid_factory.template new_grid<
                axis::open<axis::label::e_r>,
                axis::circular<axis::label::e_phi>,
                axis::open<axis::label::e_z>, axis::irregular<>,
                axis::regular<>, axis::irregular<>>(vgrid_dims, n_vgrid_bins,
                                                    {}, bin_edges);
        } else {
            m_vol_finder = vol_finder_t{args...};
        }
    }

    /// @returns access to the volume finder
    DETRAY_HOST typename detector_type::volume_finder& volume_finder() {
        return m_vol_finder;
    }

    private:
    /// Data structure that holds a volume builder for every detector volume
    volume_data_t<std::unique_ptr<volume_builder_interface<detector_type>>>
        m_volumes{};
    /// Data structure to find volumes
    typename detector_type::volume_finder m_vol_finder{};
};

}  // namespace detray
