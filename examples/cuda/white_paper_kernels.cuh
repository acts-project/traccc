#include "edm/cell.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp" 
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/jagged_vector_data.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"

void cell_test(vecmem::data::jagged_vector_view< traccc::cell > cells);

void int_test(vecmem::data::jagged_vector_view< int > data);
