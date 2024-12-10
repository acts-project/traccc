/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "detray/definitions/detail/cuda_definitions.hpp"
#include "detray/detectors/telescope_metadata.hpp"
#include "detray/detectors/toy_metadata.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/line_stepper.hpp"
#include "material_validation.hpp"

namespace detray::cuda {

template <typename detector_t>
__global__ void material_validation_kernel(
    typename detector_t::view_type det_data, const propagation::config cfg,
    vecmem::data::vector_view<
        free_track_parameters<typename detector_t::algebra_type>>
        tracks_view,
    vecmem::data::vector_view<
        material_validator::material_record<typename detector_t::scalar_type>>
        mat_records_view,
    vecmem::data::jagged_vector_view<
        material_validator::material_params<typename detector_t::scalar_type>>
        mat_steps_view) {

    using detector_device_t =
        detector<typename detector_t::metadata, device_container_types>;
    using algebra_t = typename detector_device_t::algebra_type;
    using scalar_t = dscalar<algebra_t>;

    using stepper_t = line_stepper<algebra_t>;
    using navigator_t = navigator<detector_device_t>;
    // Propagator with full covariance transport, pathlimit aborter and
    // material tracer
    using material_tracer_t =
        material_validator::material_tracer<scalar_t, vecmem::device_vector>;
    using actor_chain_t =
        actor_chain<tuple, pathlimit_aborter, parameter_transporter<algebra_t>,
                    parameter_resetter<algebra_t>,
                    pointwise_material_interactor<algebra_t>,
                    material_tracer_t>;
    using propagator_t = propagator<stepper_t, navigator_t, actor_chain_t>;

    detector_device_t det(det_data);

    vecmem::device_vector<free_track_parameters<algebra_t>> tracks(tracks_view);
    vecmem::device_vector<typename material_tracer_t::material_record_type>
        mat_records(mat_records_view);
    vecmem::jagged_device_vector<
        typename material_tracer_t::material_params_type>
        mat_steps(mat_steps_view);

    int trk_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (trk_id >= tracks.size()) {
        return;
    }

    propagator_t p{cfg};

    // Create the actor states
    pathlimit_aborter::state aborter_state{cfg.stepping.path_limit};
    typename parameter_transporter<algebra_t>::state transporter_state{};
    typename parameter_resetter<algebra_t>::state resetter_state{};
    typename pointwise_material_interactor<algebra_t>::state interactor_state{};
    typename material_tracer_t::state mat_tracer_state{mat_steps.at(trk_id)};

    auto actor_states =
        ::detray::tie(aborter_state, transporter_state, resetter_state,
                      interactor_state, mat_tracer_state);

    // Run propagation
    typename navigator_t::state::view_type nav_view{};
    typename propagator_t::state propagation(tracks[trk_id], det, nav_view);

    p.propagate(propagation, actor_states);

    // Record the accumulated material
    assert(mat_records.size() == tracks.size());
    mat_records.at(trk_id) = mat_tracer_state.get_material_record();
}

/// Launch the device kernel
template <typename detector_t>
void material_validation_device(
    typename detector_t::view_type det_view, const propagation::config &cfg,
    vecmem::data::vector_view<
        free_track_parameters<typename detector_t::algebra_type>> &tracks_view,
    vecmem::data::vector_view<
        material_validator::material_record<typename detector_t::scalar_type>>
        &mat_records_view,
    vecmem::data::jagged_vector_view<
        material_validator::material_params<typename detector_t::scalar_type>>
        &mat_steps_view) {

    constexpr int thread_dim = 2 * WARP_SIZE;
    int block_dim = tracks_view.size() / thread_dim + 1;

    // run the test kernel
    material_validation_kernel<detector_t><<<block_dim, thread_dim>>>(
        det_view, cfg, tracks_view, mat_records_view, mat_steps_view);

    // cuda error check
    DETRAY_CUDA_ERROR_CHECK(cudaGetLastError());
    DETRAY_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

/// Macro declaring the template instantiations for the different detector types
#define DECLARE_MATERIAL_VALIDATION(METADATA)                                 \
                                                                              \
    template void material_validation_device<detector<METADATA>>(             \
        typename detector<METADATA>::view_type, const propagation::config &,  \
        vecmem::data::vector_view<                                            \
            free_track_parameters<typename detector<METADATA>::algebra_type>> \
            &,                                                                \
        vecmem::data::vector_view<material_validator::material_record<        \
            typename detector<METADATA>::scalar_type>> &,                     \
        vecmem::data::jagged_vector_view<material_validator::material_params< \
            typename detector<METADATA>::scalar_type>> &);

DECLARE_MATERIAL_VALIDATION(default_metadata)
DECLARE_MATERIAL_VALIDATION(toy_metadata)
DECLARE_MATERIAL_VALIDATION(telescope_metadata<rectangle2D>)

}  // namespace detray::cuda
