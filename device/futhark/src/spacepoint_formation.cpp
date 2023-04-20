/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <sstream>
#include <traccc/edm/measurement.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/futhark/spacepoint_formation.hpp>
#include <traccc/futhark/wrapper.hpp>

namespace traccc::futhark {
struct measurements_to_spacepoints_wrapper
    : public wrapper<
          measurements_to_spacepoints_wrapper,
          std::tuple<futhark_u64_1d_wrapper, futhark_f32_1d_wrapper,
                     futhark_u64_1d_wrapper, futhark_u64_1d_wrapper,
                     futhark_f32_1d_wrapper, futhark_f32_1d_wrapper,
                     futhark_f32_1d_wrapper, futhark_f32_1d_wrapper>,
          std::tuple<futhark_u64_1d_wrapper, futhark_f32_1d_wrapper,
                     futhark_f32_1d_wrapper, futhark_f32_1d_wrapper>> {
    static constexpr auto* entry_f = &futhark_entry_measurements_to_spacepoints;
};

spacepoint_formation::spacepoint_formation(vecmem::memory_resource& mr)
    : m_mr(mr) {}

spacepoint_formation::output_type spacepoint_formation::operator()(
    const std::vector<std::pair<geometry_id, transform3>>& transforms,
    const measurement_container_types::host& data) const {
    std::size_t total_measurements = 0, total_transforms = transforms.size();
    std::vector<measurement> measurements;
    std::vector<geometry_id> geom_ids;

    for (std::size_t i = 0; i < data.size(); ++i) {
        for (std::size_t j = 0; j < data.at(i).items.size(); ++j) {
            ++total_measurements;
            measurements.push_back(data.at(i).items.at(j));
            geom_ids.push_back(data.at(i).header.module);
        }
    }

    std::vector<uint64_t> host_transform_geometry(total_transforms);
    std::vector<float> host_transform_transform(4 * 4 * total_transforms);

    std::vector<uint64_t> host_measurement_event(total_measurements);
    std::vector<uint64_t> host_measurement_geometry(total_measurements);
    std::vector<float> host_measurement_position0(total_measurements);
    std::vector<float> host_measurement_position1(total_measurements);
    std::vector<float> host_measurement_variance0(total_measurements);
    std::vector<float> host_measurement_variance1(total_measurements);

    for (std::size_t i = 0; i < total_transforms; ++i) {
        host_transform_geometry[i] = std::get<0>(transforms[i]);
        transform3 transform = std::get<1>(transforms[i]);
        transform3::element_getter getter;
        for (std::size_t x = 0; x < 4; ++x) {
            for (std::size_t y = 0; y < 4; ++y) {
                host_transform_transform[9 * i + 3 * x + y] =
                    getter(transform.matrix(), x, y);
            }
        }
    }

    for (std::size_t i = 0, k = 0; i < data.size(); ++i) {
        for (std::size_t j = 0; j < data.at(i).items.size(); ++j, ++k) {
            host_measurement_event[k] = 0;
            host_measurement_geometry[k] = geom_ids[k];
            host_measurement_position0[k] = data.at(i).items.at(j).local[0];
            host_measurement_position1[k] = data.at(i).items.at(j).local[1];
            host_measurement_variance0[k] = data.at(i).items.at(j).variance[0];
            host_measurement_variance1[k] = data.at(i).items.at(j).variance[1];
        }
    }

    measurements_to_spacepoints_wrapper::output_t r =
        measurements_to_spacepoints_wrapper::run(
            std::move(host_transform_geometry),
            std::move(host_transform_transform),
            std::move(host_measurement_event),
            std::move(host_measurement_geometry),
            std::move(host_measurement_position0),
            std::move(host_measurement_position1),
            std::move(host_measurement_variance0),
            std::move(host_measurement_variance1));

    output_type out(&m_mr);

    for (std::size_t i = 0; i < total_measurements; ++i) {
        out.push_back({std::get<1>(r)[i], std::get<2>(r)[i], std::get<3>(r)[i],
                       measurements[i]});
    }

    return out;
}
}  // namespace traccc::futhark
