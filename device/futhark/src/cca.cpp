/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <traccc/futhark/cca_core.h>

#include <sstream>
#include <traccc/futhark/cca.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

namespace traccc::futhark {
component_connection::component_connection()
    : cfg(futhark_context_config_new()), ctx(futhark_context_new(cfg)) {}

component_connection::~component_connection() {
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
}

component_connection::output_type component_connection::operator()(
    const cell_container_types::host& data) const {
    std::size_t total_cells = 0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        total_cells += data.at(i).items.size();
    }

    uint64_t* host_event = new uint64_t[total_cells];
    uint64_t* host_geometry = new uint64_t[total_cells];
    int64_t* host_channel0 = new int64_t[total_cells];
    int64_t* host_channel1 = new int64_t[total_cells];
    float* host_activation = new float[total_cells];

    for (std::size_t i = 0, k = 0; i < data.size(); ++i) {
        for (std::size_t j = 0; j < data.at(i).items.size(); ++j, ++k) {
            host_event[k] = 0;
            host_geometry[k] = data.at(i).header.module;
            host_channel0[k] = data.at(i).items.at(j).channel0;
            host_channel1[k] = data.at(i).items.at(j).channel1;
            host_activation[k] = data.at(i).items.at(j).activation;
        }
    }

    struct futhark_u64_1d* in_event =
        futhark_new_u64_1d(ctx, host_event, total_cells);
    struct futhark_u64_1d* in_geometry =
        futhark_new_u64_1d(ctx, host_geometry, total_cells);
    struct futhark_i64_1d* in_channel0 =
        futhark_new_i64_1d(ctx, host_channel0, total_cells);
    struct futhark_i64_1d* in_channel1 =
        futhark_new_i64_1d(ctx, host_channel1, total_cells);
    struct futhark_f32_1d* in_activation =
        futhark_new_f32_1d(ctx, host_activation, total_cells);

    delete[] host_event;
    delete[] host_geometry;
    delete[] host_channel0;
    delete[] host_channel1;
    delete[] host_activation;

    struct futhark_u64_1d* out_event;
    struct futhark_u64_1d* out_geometry;
    struct futhark_f32_1d* out_position0;
    struct futhark_f32_1d* out_position1;
    struct futhark_f32_1d* out_variance0;
    struct futhark_f32_1d* out_variance1;

    int r = futhark_entry_cells_to_measurements_entry(
        ctx, &out_event, &out_geometry, &out_position0, &out_position1,
        &out_variance0, &out_variance1, in_event, in_geometry, in_channel0,
        in_channel1, in_activation);

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

    futhark_free_u64_1d(ctx, in_event);
    futhark_free_u64_1d(ctx, in_geometry);
    futhark_free_i64_1d(ctx, in_channel0);
    futhark_free_i64_1d(ctx, in_channel1);
    futhark_free_f32_1d(ctx, in_activation);

    int64_t total_measurements = *futhark_shape_u64_1d(ctx, out_event);

    uint64_t* host_out_event = new uint64_t[total_measurements];
    uint64_t* host_out_geometry = new uint64_t[total_measurements];
    float* host_out_position0 = new float[total_measurements];
    float* host_out_position1 = new float[total_measurements];
    float* host_out_variance0 = new float[total_measurements];
    float* host_out_variance1 = new float[total_measurements];

    futhark_values_u64_1d(ctx, out_event, host_out_event);
    futhark_values_u64_1d(ctx, out_geometry, host_out_geometry);
    futhark_values_f32_1d(ctx, out_position0, host_out_position0);
    futhark_values_f32_1d(ctx, out_position1, host_out_position1);
    futhark_values_f32_1d(ctx, out_variance0, host_out_variance0);
    futhark_values_f32_1d(ctx, out_variance1, host_out_variance1);

    futhark_free_u64_1d(ctx, out_event);
    futhark_free_u64_1d(ctx, out_geometry);
    futhark_free_f32_1d(ctx, out_position0);
    futhark_free_f32_1d(ctx, out_position1);
    futhark_free_f32_1d(ctx, out_variance0);
    futhark_free_f32_1d(ctx, out_variance1);

    output_type out;
    vecmem::host_memory_resource mem;

    for (std::size_t i = 0; i < data.size(); ++i) {
        vecmem::vector<measurement> v(&mem);

        for (std::size_t j = 0; j < total_measurements; ++j) {
            if (host_out_geometry[j] == data.at(i).header.module) {
                measurement m;

                m.local = {host_out_position0[j], host_out_position1[j]};
                m.variance = {host_out_variance0[j], host_out_variance1[j]};

                v.push_back(m);
            }
        }

        out.push_back(cell_module(data.at(i).header), std::move(v));
    }

    delete[] host_out_event;
    delete[] host_out_geometry;
    delete[] host_out_position0;
    delete[] host_out_position1;
    delete[] host_out_variance0;
    delete[] host_out_variance1;

    return out;
}
}  // namespace traccc::futhark
