-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

import "linear"
import "edm"
import "zip"

import "measurement_creation"

entry cells_to_measurements [n]
    (es: [n]u64) (gs: [n]u64) (c0s: [n]i64) (c1s: [n]i64) (as: [n]f32):
    ([]u64, []u64, []f32, []f32, []f32, []f32) =
    (zip5 es gs c0s c1s as) |>
    map (\(e, g, c0, c1, a) ->
         {event=e, geometry=g, position=(c0, c1), activation=a}) >->
    cells_to_measurements_impl |>
    map (\(x: Measurement) ->
         (x.event, x.geometry, x.position.0, x.position.1,
          x.variance.0, x.variance.1)) >->
    unzip6
