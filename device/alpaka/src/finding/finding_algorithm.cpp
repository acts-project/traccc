/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/finding/finding_algorithm.hpp"

#include "../utils/barrier.hpp"
#include "../utils/thread_id.hpp"
#include "../utils/utils.hpp"
#include "./kernels/apply_interaction.hpp"
#include "./kernels/build_tracks.hpp"
#include "./kernels/fill_sort_keys.hpp"
#include "./kernels/find_tracks.hpp"
#include "./kernels/make_barcode_sequence.hpp"
#include "./kernels/propagate_to_next_surface.hpp"
#include "./kernels/prune_tracks.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/detector_type_utils.hpp"
#include "traccc/utils/projections.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/unique_ptr.hpp>

// Thrust include(s).
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// System include(s).
#include <cassert>
#include <vector>

namespace traccc::alpaka {

template <typename stepper_t, typename navigator_t>
finding_algorithm<stepper_t, navigator_t>::finding_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_cfg(cfg), m_mr(mr), m_copy(copy) {}

template <typename stepper_t, typename navigator_t>
track_candidate_container_types::buffer
finding_algorithm<stepper_t, navigator_t>::operator()(
    const typename detector_type::view_type& det_view,
    const bfield_type& field_view,
    const typename measurement_collection_types::view& measurements,
    const bound_track_parameters_collection_types::buffer& seeds_buffer) const {

    assert(m_cfg.min_step_length_for_next_surface >
               math::fabs(m_cfg.propagation.navigation.overstep_tolerance) &&
           "Min step length for the next surface should be higher than the "
           "overstep tolerance");

    // Setup alpaka
    auto devHost = ::alpaka::getDevByIdx(::alpaka::Platform<Host>{}, 0u);
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = Queue{devAcc};
    Idx threadsPerBlock = getWarpSize<Acc>() * 2;

    // Copy setup
    m_copy.setup(seeds_buffer)->ignore();

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto thrustExecPolicy = thrust::device;
#else
    auto thrustExecPolicy = thrust::host;
#endif

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    unsigned int n_modules;
    measurement_collection_types::const_view::size_type n_measurements =
        m_copy.get_size(measurements);

    // Get copy of barcode uniques
    measurement_collection_types::buffer uniques_buffer{n_measurements,
                                                        m_mr.main};
    m_copy.setup(uniques_buffer)->ignore();

    {
        measurement_collection_types::device uniques(uniques_buffer);

        measurement* uniques_end =
            thrust::unique_copy(thrustExecPolicy, measurements.ptr(),
                                measurements.ptr() + n_measurements,
                                uniques.begin(), measurement_equal_comp());
        n_modules = static_cast<unsigned int>(uniques_end - uniques.begin());
    }

    // Get upper bounds of unique elements
    vecmem::data::vector_buffer<unsigned int> upper_bounds_buffer{n_modules,
                                                                  m_mr.main};
    m_copy.setup(upper_bounds_buffer)->ignore();

    {
        vecmem::device_vector<unsigned int> upper_bounds(upper_bounds_buffer);

        measurement_collection_types::device uniques(uniques_buffer);

        thrust::upper_bound(thrustExecPolicy, measurements.ptr(),
                            measurements.ptr() + n_measurements,
                            uniques.begin(), uniques.begin() + n_modules,
                            upper_bounds.begin(), measurement_sort_comp());
    }

    /*****************************************************************
     * Kernel1: Create barcode sequence
     *****************************************************************/

    vecmem::data::vector_buffer<detray::geometry::barcode> barcodes_buffer{
        n_modules, m_mr.main};
    m_copy.setup(barcodes_buffer)->ignore();

    {
        Idx blocksPerGrid =
            (barcodes_buffer.size() + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        ::alpaka::exec<Acc>(queue, workDiv, MakeBarcodeSequenceKernel{},
                            device::make_barcode_sequence_payload{
                                vecmem::get_data(uniques_buffer),
                                vecmem::get_data(barcodes_buffer)});
        ::alpaka::wait(queue);
    }

    const unsigned int n_seeds = m_copy.get_size(seeds_buffer);

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(n_seeds,
                                                                     m_mr.main);
    m_copy.setup(in_params_buffer)->ignore();
    m_copy(vecmem::get_data(seeds_buffer), vecmem::get_data(in_params_buffer))
        ->ignore();
    vecmem::data::vector_buffer<unsigned int> param_liveness_buffer(n_seeds,
                                                                    m_mr.main);
    m_copy.setup(param_liveness_buffer)->ignore();
    m_copy.memset(param_liveness_buffer, 1)->ignore();

    // Number of tracks per seed
    vecmem::data::vector_buffer<unsigned int> n_tracks_per_seed_buffer(
        n_seeds, m_mr.main);
    m_copy.setup(n_tracks_per_seed_buffer)->ignore();

    // Create a buffer for links
    unsigned int link_buffer_capacity = m_cfg.initial_links_per_seed * n_seeds;
    vecmem::data::vector_buffer<candidate_link> links_buffer(
        link_buffer_capacity, m_mr.main, vecmem::data::buffer_type::resizable);
    m_copy.setup(links_buffer)->wait();

    // Create a buffer of tip links
    vecmem::data::vector_buffer<unsigned int> tips_buffer{
        m_cfg.max_num_branches_per_seed * n_seeds, m_mr.main,
        vecmem::data::buffer_type::resizable};
    m_copy.setup(tips_buffer)->wait();

    std::map<unsigned int, unsigned int> step_to_link_idx_map;
    step_to_link_idx_map[0] = 0;

    unsigned int n_in_params = n_seeds;

    for (unsigned int step = 0;
         step < m_cfg.max_track_candidates_per_track && n_in_params > 0;
         step++) {

        /*****************************************************************
         * Kernel2: Apply material interaction
         ****************************************************************/

        {
            Idx blocksPerGrid =
                (n_in_params + threadsPerBlock - 1) / threadsPerBlock;
            auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

            ::alpaka::exec<Acc>(
                queue, workDiv,
                ApplyInteractionKernel<std::decay_t<detector_type>>{}, m_cfg,
                device::apply_interaction_payload<std::decay_t<detector_type>>{
                    det_view, n_in_params, vecmem::get_data(in_params_buffer),
                    vecmem::get_data(param_liveness_buffer)});
            ::alpaka::wait(queue);
        }

        /*****************************************************************
         * Kernel3: Count the number of measurements per parameter
         ****************************************************************/

        unsigned int n_candidates = 0;

        {
            // Buffer for kalman-updated parameters spawned by the measurement
            // candidates
            const unsigned int n_max_candidates =
                n_in_params * m_cfg.max_num_branches_per_surface;

            bound_track_parameters_collection_types::buffer
                updated_params_buffer(
                    n_in_params * m_cfg.max_num_branches_per_surface,
                    m_mr.main);
            m_copy.setup(updated_params_buffer)->ignore();

            vecmem::data::vector_buffer<unsigned int> updated_liveness_buffer(
                n_in_params * m_cfg.max_num_branches_per_surface, m_mr.main);
            m_copy.setup(updated_liveness_buffer)->ignore();

            const unsigned int links_size = m_copy.get_size(links_buffer);

            if (links_size + n_max_candidates > link_buffer_capacity) {
                const unsigned int new_link_buffer_capacity = std::max(
                    2 * link_buffer_capacity, links_size + n_max_candidates);

                TRACCC_INFO("Link buffer (capacity "
                            << link_buffer_capacity << ") is too small to hold "
                            << links_size << " current and " << n_max_candidates
                            << " new links; increasing capacity to "
                            << new_link_buffer_capacity);

                link_buffer_capacity = new_link_buffer_capacity;

                vecmem::data::vector_buffer<candidate_link> new_links_buffer(
                    link_buffer_capacity, m_mr.main,
                    vecmem::data::buffer_type::resizable);

                m_copy.setup(new_links_buffer)->wait();
                m_copy(links_buffer, new_links_buffer)->wait();

                links_buffer = std::move(new_links_buffer);
            }

            Idx blocksPerGrid =
                (n_in_params + threadsPerBlock - 1) / threadsPerBlock;
            auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

            const unsigned int prev_link_idx =
                step == 0 ? 0 : step_to_link_idx_map[step - 1];

            assert(links_size == step_to_link_idx_map[step]);

            typedef device::find_tracks_payload<std::decay_t<detector_type>>
                PayloadType;

            auto bufHost_payload =
                ::alpaka::allocBuf<PayloadType, Idx>(devHost, 1u);
            PayloadType* payload = ::alpaka::getPtrNative(bufHost_payload);

            new (payload) PayloadType{
                .det_data = det_view,
                .measurements_view = measurements,
                .in_params_view = vecmem::get_data(in_params_buffer),
                .in_params_liveness_view =
                    vecmem::get_data(param_liveness_buffer),
                .n_in_params = n_in_params,
                .barcodes_view = vecmem::get_data(barcodes_buffer),
                .upper_bounds_view = vecmem::get_data(upper_bounds_buffer),
                .links_view = vecmem::get_data(links_buffer),
                .prev_links_idx = prev_link_idx,
                .curr_links_idx = step_to_link_idx_map[step],
                .step = step,
                .out_params_view = vecmem::get_data(updated_params_buffer),
                .out_params_liveness_view =
                    vecmem::get_data(updated_liveness_buffer)};

            auto bufAcc_payload =
                ::alpaka::allocBuf<PayloadType, Idx>(devAcc, 1u);
            ::alpaka::memcpy(queue, bufAcc_payload, bufHost_payload);
            ::alpaka::wait(queue);

            ::alpaka::exec<Acc>(queue, workDiv,
                                FindTracksKernel<std::decay_t<detector_type>>{},
                                m_cfg, ::alpaka::getPtrNative(bufAcc_payload));
            ::alpaka::wait(queue);

            std::swap(in_params_buffer, updated_params_buffer);
            std::swap(param_liveness_buffer, updated_liveness_buffer);

            // Create a buffer for links
            step_to_link_idx_map[step + 1] = m_copy.get_size(links_buffer);
            n_candidates =
                step_to_link_idx_map[step + 1] - step_to_link_idx_map[step];
            ::alpaka::wait(queue);
        }

        if (n_candidates > 0) {
            /*****************************************************************
             * Kernel4: Get key and value for parameter sorting
             *****************************************************************/

            vecmem::data::vector_buffer<unsigned int> param_ids_buffer(
                n_candidates, m_mr.main);
            m_copy.setup(param_ids_buffer)->ignore();

            {
                vecmem::data::vector_buffer<device::sort_key> keys_buffer(
                    n_candidates, m_mr.main);
                m_copy.setup(keys_buffer)->ignore();

                Idx blocksPerGrid =
                    (n_candidates + threadsPerBlock - 1) / threadsPerBlock;
                auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

                ::alpaka::exec<Acc>(queue, workDiv, FillSortKeysKernel{},
                                    device::fill_sort_keys_payload{
                                        vecmem::get_data(in_params_buffer),
                                        vecmem::get_data(keys_buffer),
                                        vecmem::get_data(param_ids_buffer)});
                ::alpaka::wait(queue);

                // Sort the key and values
                vecmem::device_vector<device::sort_key> keys_device(
                    keys_buffer);
                vecmem::device_vector<unsigned int> param_ids_device(
                    param_ids_buffer);
                thrust::sort_by_key(thrustExecPolicy, keys_device.begin(),
                                    keys_device.end(),
                                    param_ids_device.begin());
            }

            /*****************************************************************
             * Kernel5: Propagate to the next surface
             *****************************************************************/

            {
                // Reset the number of tracks per seed
                m_copy.memset(n_tracks_per_seed_buffer, 0)->ignore();

                Idx blocksPerGrid =
                    (n_candidates + threadsPerBlock - 1) / threadsPerBlock;
                auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

                typedef device::propagate_to_next_surface_payload<
                    std::decay_t<propagator_type>, std::decay_t<bfield_type>>
                    PayloadType;

                auto bufHost_payload =
                    ::alpaka::allocBuf<PayloadType, Idx>(devHost, 1u);
                PayloadType* payload = ::alpaka::getPtrNative(bufHost_payload);

                new (payload) PayloadType{
                    .det_data = det_view,
                    .field_data = field_view,
                    .params_view = vecmem::get_data(in_params_buffer),
                    .params_liveness_view =
                        vecmem::get_data(param_liveness_buffer),
                    .param_ids_view = vecmem::get_data(param_ids_buffer),
                    .links_view = vecmem::get_data(links_buffer),
                    .prev_links_idx = step_to_link_idx_map[step],
                    .step = step,
                    .n_in_params = n_candidates,
                    .tips_view = vecmem::get_data(tips_buffer),
                    .n_tracks_per_seed_view =
                        vecmem::get_data(n_tracks_per_seed_buffer)};

                auto bufAcc_payload =
                    ::alpaka::allocBuf<PayloadType, Idx>(devAcc, 1u);
                ::alpaka::memcpy(queue, bufAcc_payload, bufHost_payload);
                ::alpaka::wait(queue);

                ::alpaka::exec<Acc>(
                    queue, workDiv,
                    PropagateToNextSurfaceKernel<std::decay_t<propagator_type>,
                                                 std::decay_t<bfield_type>>{},
                    m_cfg, ::alpaka::getPtrNative(bufAcc_payload));
                ::alpaka::wait(queue);
            }
        }

        n_in_params = n_candidates;
    }

    TRACCC_DEBUG(
        "Final link buffer usage was "
        << m_copy.get_size(links_buffer) << " out of " << link_buffer_capacity
        << " ("
        << ((100.f * static_cast<float>(m_copy.get_size(links_buffer))) /
            static_cast<float>(link_buffer_capacity))
        << "%)");

    /*****************************************************************
     * Kernel6: Build tracks
     *****************************************************************/

    // Get the number of tips
    auto n_tips_total = m_copy.get_size(tips_buffer);

    // Create track candidate buffer
    track_candidate_container_types::buffer track_candidates_buffer{
        {n_tips_total, m_mr.main},
        {std::vector<std::size_t>(n_tips_total,
                                  m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};

    m_copy.setup(track_candidates_buffer.headers)->ignore();
    m_copy.setup(track_candidates_buffer.items)->ignore();

    // Create buffer for valid indices
    vecmem::data::vector_buffer<unsigned int> valid_indices_buffer(n_tips_total,
                                                                   m_mr.main);

    // Count the number of valid tracks
    auto bufHost_n_valid_tracks =
        ::alpaka::allocBuf<unsigned int, Idx>(devHost, 1u);
    unsigned int* n_valid_tracks =
        ::alpaka::getPtrNative(bufHost_n_valid_tracks);
    ::alpaka::memset(queue, bufHost_n_valid_tracks, 0);
    ::alpaka::wait(queue);

    // @Note: nBlocks can be zero in case there is no tip. This happens when
    // chi2_max config is set tightly and no tips are found
    if (n_tips_total > 0) {
        auto n_valid_tracks_device =
            ::alpaka::allocBuf<unsigned int, Idx>(devAcc, 1u);
        ::alpaka::memset(queue, n_valid_tracks_device, 0);

        Idx blocksPerGrid =
            (n_tips_total + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        track_candidate_container_types::view track_candidates_view(
            track_candidates_buffer);

        ::alpaka::exec<Acc>(
            queue, workDiv, BuildTracksKernel{}, m_cfg,
            device::build_tracks_payload{
                measurements, vecmem::get_data(seeds_buffer),
                vecmem::get_data(links_buffer), vecmem::get_data(tips_buffer),
                track_candidates_view, vecmem::get_data(valid_indices_buffer),
                ::alpaka::getPtrNative(n_valid_tracks_device)});
        ::alpaka::wait(queue);

        // Global counter object: Device -> Host
        ::alpaka::memcpy(queue, bufHost_n_valid_tracks, n_valid_tracks_device);
        ::alpaka::wait(queue);
    }

    // Create pruned candidate buffer
    track_candidate_container_types::buffer prune_candidates_buffer{
        {*n_valid_tracks, m_mr.main},
        {std::vector<std::size_t>(*n_valid_tracks,
                                  m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};

    m_copy.setup(prune_candidates_buffer.headers)->ignore();
    m_copy.setup(prune_candidates_buffer.items)->ignore();

    if (*n_valid_tracks > 0) {
        Idx blocksPerGrid =
            (*n_valid_tracks + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        track_candidate_container_types::const_view track_candidates_view(
            track_candidates_buffer);

        track_candidate_container_types::view prune_candidates_view(
            prune_candidates_buffer);

        ::alpaka::exec<Acc>(
            queue, workDiv, PruneTracksKernel{},
            device::prune_tracks_payload{track_candidates_view,
                                         vecmem::get_data(valid_indices_buffer),
                                         prune_candidates_view});
        ::alpaka::wait(queue);
    }

    return prune_candidates_buffer;
}

// Explicit template instantiation
template class finding_algorithm<
    ::traccc::stepper_for_t<::traccc::default_detector::device>,
    ::traccc::navigator_for_t<::traccc::default_detector::device>>;

}  // namespace traccc::alpaka

// Add an Alpaka trait that the measurement_collection_types::const_device type
// is trivially copyable
namespace alpaka {

template <>
struct IsKernelArgumentTriviallyCopyable<
    traccc::measurement_collection_types::const_device> : std::true_type {};

}  // namespace alpaka

// Also need to add an Alpaka trait for the dynamic shared memory
namespace alpaka::trait {

template <typename TAcc, typename detector_t>
struct BlockSharedMemDynSizeBytes<traccc::alpaka::FindTracksKernel<detector_t>,
                                  TAcc> {
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
        traccc::alpaka::FindTracksKernel<detector_t> const& /* kernel */,
        TVec const& blockThreadExtent, TVec const& /* threadElemExtent */,
        TArgs const&... /* args */
        ) -> std::size_t {
        return static_cast<std::size_t>(blockThreadExtent.prod()) *
                   sizeof(unsigned int) +
               2 * static_cast<std::size_t>(blockThreadExtent.prod()) *
                   sizeof(std::pair<unsigned int, unsigned int>);
    }
};

}  // namespace alpaka::trait
