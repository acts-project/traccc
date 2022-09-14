-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

type Point2 = (f32, f32)

type Point3 = (f32, f32, f32)

type Affine3 = [4][4]f32

def transform (t: Affine3) ((x, y): Point2): Point3 = (
    x * t[0, 0] + y * t[0, 1] + t[0, 3],
    x * t[1, 0] + y * t[1, 1] + t[1, 3],
    x * t[2, 0] + y * t[2, 1] + t[2, 3]
)
