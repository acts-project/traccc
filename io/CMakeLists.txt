# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Look for OpenMP.
find_package( OpenMP COMPONENTS CXX )

# Set up the "build" of the traccc::io library.
traccc_add_library( traccc_io io TYPE SHARED
  # Public headers
  "include/traccc/io/read.hpp"
  "include/traccc/io/read_cells.hpp"
  "include/traccc/io/read_digitization_config.hpp"
  "include/traccc/io/read_geometry.hpp"
  "include/traccc/io/read_measurements.hpp"
  "include/traccc/io/read_particles.hpp"
  "include/traccc/io/read_spacepoints.hpp"
  "include/traccc/io/read_spacepoints_alt.hpp"
  "include/traccc/io/data_format.hpp"
  "include/traccc/io/event_map.hpp"
  "include/traccc/io/event_map2.hpp"
  "include/traccc/io/demonstrator_alt_edm.hpp"
  "include/traccc/io/mapper.hpp"
  "include/traccc/io/write.hpp"
  "include/traccc/io/utils.hpp"
  "include/traccc/io/details/read_surfaces.hpp"
  "include/traccc/io/reader_edm.hpp"
  # Implementation
  "src/data_format.cpp"
  "src/event_map2.cpp"
  "src/mapper.cpp"
  "src/read.cpp"
  "src/read_cells.cpp"
  "src/read_digitization_config.cpp"
  "src/read_geometry.cpp"
  "src/read_measurements.cpp"
  "src/read_particles.cpp"
  "src/read_spacepoints.cpp"
  "src/read_spacepoints_alt.cpp"
  "src/write.cpp"
  "src/utils.cpp"
  "src/read_binary.hpp"
  "src/write_binary.hpp"
  "src/details/read_surfaces.cpp"
  "src/csv/surface.hpp"
  "src/csv/make_surface_reader.hpp"
  "src/csv/make_surface_reader.cpp"
  "src/csv/read_surfaces.hpp"
  "src/csv/read_surfaces.cpp"
  "src/csv/cell.hpp"
  "src/csv/make_cell_reader.hpp"
  "src/csv/make_cell_reader.cpp"
  "src/csv/read_cells.hpp"
  "src/csv/read_cells.cpp"
  "src/csv/hit.hpp"
  "src/csv/make_hit_reader.hpp"
  "src/csv/make_hit_reader.cpp"
  "src/csv/read_spacepoints.hpp"
  "src/csv/read_spacepoints.cpp"
  "src/csv/read_spacepoints_alt.hpp"
  "src/csv/read_spacepoints_alt.cpp"
  "src/csv/measurement.hpp"
  "src/csv/make_measurement_reader.hpp"
  "src/csv/make_measurement_reader.cpp"
  "src/csv/read_measurements.hpp"
  "src/csv/read_measurements.cpp"
  "src/csv/measurement_hit_id.hpp"
  "src/csv/make_measurement_hit_id_reader.hpp"
  "src/csv/make_measurement_hit_id_reader.cpp"
  "src/csv/particle.hpp"
  "src/csv/make_particle_reader.hpp"
  "src/csv/make_particle_reader.cpp"
  "src/csv/read_particles.hpp"
  "src/csv/read_particles.cpp" )
target_link_libraries( traccc_io
  PUBLIC vecmem::core traccc::core
  PRIVATE dfelibs::dfelibs ActsPluginJson )
target_compile_definitions( traccc_io
  PRIVATE TRACCC_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data" )
if( OpenMP_CXX_FOUND )
  target_link_libraries( traccc_io PRIVATE OpenMP::OpenMP_CXX )
endif()
