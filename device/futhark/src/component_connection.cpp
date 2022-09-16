/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <memory>
#include <traccc/futhark/component_connection.hpp>
#include <traccc/futhark/utils.hpp>
#include <traccc/futhark/wrapper.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vector>

namespace traccc::futhark {
struct cells_to_measurements_wrapper
    : public wrapper<
          cells_to_measurements_wrapper,
          std::tuple<futhark_u64_1d_wrapper, futhark_u64_1d_wrapper,
                     futhark_i64_1d_wrapper, futhark_i64_1d_wrapper,
                     futhark_f32_1d_wrapper>,
          std::tuple<futhark_u64_1d_wrapper, futhark_u64_1d_wrapper,
                     futhark_f32_1d_wrapper, futhark_f32_1d_wrapper,
                     futhark_f32_1d_wrapper, futhark_f32_1d_wrapper>> {
    static constexpr auto *entry_f = &futhark_entry_cells_to_measurements;
};

component_connection::component_connection(vecmem::memory_resource &mr)
    : m_mr(mr) {}

component_connection::output_type component_connection::operator()(
    const cell_container_types::host &data) const {
    std::size_t total_cells = 0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        total_cells += data.at(i).items.size();
    }

    std::vector<uint64_t> host_event(total_cells);
    std::vector<uint64_t> host_geometry(total_cells);
    std::vector<int64_t> host_channel0(total_cells);
    std::vector<int64_t> host_channel1(total_cells);
    std::vector<float> host_activation(total_cells);

    for (std::size_t i = 0, k = 0; i < data.size(); ++i) {
        for (std::size_t j = 0; j < data.at(i).items.size(); ++j, ++k) {
            host_event[k] = 0;
            host_geometry[k] = data.at(i).header.module;
            host_channel0[k] = data.at(i).items.at(j).channel0;
            host_channel1[k] = data.at(i).items.at(j).channel1;
            host_activation[k] = data.at(i).items.at(j).activation;
        }
    }

    cells_to_measurements_wrapper::output_t r =
        cells_to_measurements_wrapper::run(
            std::move(host_event), std::move(host_geometry),
            std::move(host_channel0), std::move(host_channel1),
            std::move(host_activation));

    output_type out(&m_mr);

    for (std::size_t i = 0; i < data.size(); ++i) {
        vecmem::vector<measurement> v(&m_mr);
        v.reserve(data.at(i).items.size());

        for (std::size_t j = 0; j < std::get<1>(r).size(); ++j) {
            if (std::get<1>(r)[j] == data.at(i).header.module) {
                measurement m;

                m.local = {std::get<2>(r)[j], std::get<3>(r)[j]};
                m.variance = {std::get<4>(r)[j], std::get<5>(r)[j]};

                v.push_back(m);
            }
        }

        out.push_back(cell_module(data.at(i).header), std::move(v));
    }

    return out;
}
}  // namespace traccc::futhark
