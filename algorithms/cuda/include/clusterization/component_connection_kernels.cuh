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
#include "../../cuda/src/utils/cuda_error_handling.hpp"

#include "edm/cell.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "definitions/algebra.hpp"
#include "definitions/primitives.hpp"

namespace traccc {
  void sparse_ccl_cuda(const vecmem::data::jagged_vector_view< cell >& cell_per_event,
		       vecmem::data::jagged_vector_view< unsigned int >& label_per_event,
		       vecmem::data::vector_view<unsigned int> num_labels);
  
  void sp_formation_cuda(const vecmem::data::jagged_vector_view< cell >& cell_per_event,
			 const vecmem::data::jagged_vector_view< unsigned int >& label_per_event,
			 const vecmem::data::vector_view<unsigned int> num_labels,
			 vecmem::data::jagged_vector_view< measurement >& ms_per_event,
			 vecmem::data::jagged_vector_view< spacepoint >& sp_per_event);
}
