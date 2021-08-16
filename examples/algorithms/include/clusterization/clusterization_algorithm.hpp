/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/internal_spacepoint.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "geometry/pixel_segmentation.hpp"
#include "io/csv.hpp"
#include "io/reader.hpp"
#include "io/utils.hpp"

// clusterization
#include "clusterization/component_connection.hpp"
#include "clusterization/measurement_creation.hpp"
#include "clusterization/spacepoint_formation.hpp"

namespace traccc{

class clusterization_algorithm
    : public algorithm< const host_cell_container&, std::pair<host_measurement_container, host_spacepoint_container> > {

    struct config{


    };

    output_type operator()(const input_type& cells_per_event) const override {
	output_type o;
	this->operator()(cells_per_event, o);
	return o;
    }

    void operator()(const input_type cells_per_event, output_type& o) const override{
        // output containers
	host_measurement_container measurements_per_event = o.first;
	host_spacepoint_container spacepoints_per_event = o.second;

	// reserve the vector size
        measurements_per_event.headers.reserve(cells_per_event.headers.size());
        measurements_per_event.items.reserve(cells_per_event.headers.size());
        spacepoints_per_event.headers.reserve(cells_per_event.headers.size());
        spacepoints_per_event.items.reserve(cells_per_event.headers.size());


    }
    
private:
    // algorithms
    traccc::component_connection cc;
    traccc::measurement_creation mt;
    traccc::spacepoint_formation sp;

    
};
    
} // namespace traccc
