-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

import "linear"

type Cell = {
    event: u64,
    geometry: u64,
    position: (i64, i64),
    activation: f32
}

type Measurement = {
    event: u64,
    geometry: u64,
    position: Point2,
    variance: Point2
}

type Spacepoint = {
    event: u64,
    position: Point3
}

type Seed = {
    event: u64,
    lo: Point3,
    mi: Point3,
    hi: Point3
}
