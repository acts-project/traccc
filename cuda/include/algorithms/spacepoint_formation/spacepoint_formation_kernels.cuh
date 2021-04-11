/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "vecmem/containers/const_device_array.hpp"
#include "vecmem/containers/const_device_vector.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/jagged_vector_data.hpp"

#include "edm/cluster.hpp"
#include "cuda/include/edm/cell_container.hpp"
#include "cuda/include/edm/label_container.hpp"
#include "cuda/include/edm/measurement_container.hpp"
#include "cuda/include/edm/spacepoint_container.hpp"
#include "definitions/algebra.hpp"
#include "definitions/primitives.hpp"


namespace traccc {
    void sp_formation_cuda(traccc::cell_container_cuda& cells_per_event,
			   traccc::label_container_cuda& labels_per_event,
			   traccc::measurement_container_cuda& measurements_per_event,
			   traccc::spacepoint_container_cuda& spacepoints_per_event);
}
