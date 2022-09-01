/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <traccc/futhark/spc_core.h>

#include <sstream>
#include <traccc/edm/measurement.hpp>
#include <traccc/edm/spacepoint.hpp>
#include <traccc/futhark/spc.hpp>

namespace traccc::futhark {
spacepoint_creation::spacepoint_creation()
    : cfg(futhark_context_config_new()), ctx(futhark_context_new(cfg)) {}

spacepoint_creation::~spacepoint_creation() {
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
}

spacepoint_creation::output_type spacepoint_creation::operator()(
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

    uint64_t* host_in_transform_geometry = new uint64_t[total_transforms];
    float* host_in_transform_transform = new float[4 * 4 * total_transforms];

    uint64_t* host_in_measurement_event = new uint64_t[total_measurements];
    uint64_t* host_in_measurement_geometry = new uint64_t[total_measurements];
    float* host_in_measurement_position0 = new float[total_measurements];
    float* host_in_measurement_position1 = new float[total_measurements];
    float* host_in_measurement_variance0 = new float[total_measurements];
    float* host_in_measurement_variance1 = new float[total_measurements];

    for (std::size_t i = 0; i < total_transforms; ++i) {
        host_in_transform_geometry[i] = std::get<0>(transforms[i]);
        transform3 transform = std::get<1>(transforms[i]);
        transform3::element_getter getter;
        for (std::size_t x = 0; x < 4; ++x) {
            for (std::size_t y = 0; y < 4; ++y) {
                host_in_transform_transform[9 * i + 3 * x + y] =
                    getter(transform.matrix(), x, y);
            }
        }
    }

    for (std::size_t i = 0, k = 0; i < data.size(); ++i) {
        for (std::size_t j = 0; j < data.at(i).items.size(); ++j, ++k) {
            host_in_measurement_event[k] = 0;
            host_in_measurement_geometry[k] = geom_ids[k];
            host_in_measurement_position0[k] = data.at(i).items.at(j).local[0];
            host_in_measurement_position1[k] = data.at(i).items.at(j).local[1];
            host_in_measurement_variance0[k] =
                data.at(i).items.at(j).variance[0];
            host_in_measurement_variance1[k] =
                data.at(i).items.at(j).variance[1];
        }
    }

    struct futhark_u64_1d* in_transform_geometry =
        futhark_new_u64_1d(ctx, host_in_transform_geometry, total_transforms);
    struct futhark_f32_3d* in_transform_transform = futhark_new_f32_3d(
        ctx, host_in_transform_transform, total_transforms, 4, 4);

    struct futhark_u64_1d* in_measurement_event =
        futhark_new_u64_1d(ctx, host_in_measurement_event, total_measurements);
    struct futhark_u64_1d* in_measurement_geometry = futhark_new_u64_1d(
        ctx, host_in_measurement_geometry, total_measurements);
    struct futhark_f32_1d* in_measurement_position0 = futhark_new_f32_1d(
        ctx, host_in_measurement_position0, total_measurements);
    struct futhark_f32_1d* in_measurement_position1 = futhark_new_f32_1d(
        ctx, host_in_measurement_position1, total_measurements);
    struct futhark_f32_1d* in_measurement_variance0 = futhark_new_f32_1d(
        ctx, host_in_measurement_variance0, total_measurements);
    struct futhark_f32_1d* in_measurement_variance1 = futhark_new_f32_1d(
        ctx, host_in_measurement_variance1, total_measurements);

    delete[] host_in_transform_geometry;
    delete[] host_in_transform_transform;
    delete[] host_in_measurement_event;
    delete[] host_in_measurement_geometry;
    delete[] host_in_measurement_position0;
    delete[] host_in_measurement_position1;
    delete[] host_in_measurement_variance0;
    delete[] host_in_measurement_variance1;

    struct futhark_f32_1d* out_position0;
    struct futhark_f32_1d* out_position1;
    struct futhark_f32_1d* out_position2;

    int r = futhark_entry_measurements_to_spacepoints_entry(
        ctx, &out_position0, &out_position1, &out_position2,
        in_transform_geometry, in_transform_transform, in_measurement_event,
        in_measurement_geometry, in_measurement_position0,
        in_measurement_position1, in_measurement_variance0,
        in_measurement_variance1);

    if (r == FUTHARK_PROGRAM_ERROR) {
        throw std::runtime_error(
            "Futhark program exited due to a programming error.");
    } else if (r == FUTHARK_OUT_OF_MEMORY) {
        throw std::runtime_error(
            "Futhark program exited due to lack of allocatable memory.");
    } else if (r != FUTHARK_SUCCESS) {
        std::stringstream ss;
        ss << "Futhark program exited with unknown non-zero return code " << r
           << ".";

        throw std::runtime_error(ss.str());
    }

    futhark_free_u64_1d(ctx, in_transform_geometry);
    futhark_free_f32_3d(ctx, in_transform_transform);
    futhark_free_u64_1d(ctx, in_measurement_event);
    futhark_free_u64_1d(ctx, in_measurement_geometry);
    futhark_free_f32_1d(ctx, in_measurement_position0);
    futhark_free_f32_1d(ctx, in_measurement_position1);
    futhark_free_f32_1d(ctx, in_measurement_variance0);
    futhark_free_f32_1d(ctx, in_measurement_variance1);

    float* host_out_position0 = new float[total_measurements];
    float* host_out_position1 = new float[total_measurements];
    float* host_out_position2 = new float[total_measurements];

    futhark_values_f32_1d(ctx, out_position0, host_out_position0);
    futhark_values_f32_1d(ctx, out_position1, host_out_position1);
    futhark_values_f32_1d(ctx, out_position2, host_out_position2);

    futhark_free_f32_1d(ctx, out_position0);
    futhark_free_f32_1d(ctx, out_position1);
    futhark_free_f32_1d(ctx, out_position2);

    output_type out;

    for (std::size_t i = 0; i < total_measurements; ++i) {
        out.push_back({host_out_position0[i], host_out_position1[i],
                       host_out_position2[i], measurements[i]});
    }

    delete[] host_out_position0;
    delete[] host_out_position1;
    delete[] host_out_position2;

    return out;
}
}  // namespace traccc::futhark
