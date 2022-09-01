-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

-- The implementation corresponds to Algorithm 2 of the following paper:
-- https://epubs.siam.org/doi/pdf/10.1137/1.9781611976137.5

import "types"

type maybe_idx = #just i64 | #nothing
type maybe_pair = #just (i64, i64) | #nothing

def gather 'a (as: []a) (is: []i64) = map (\i -> as[i]) is

def from_maybe_pair (p: maybe_pair): (i64, i64) =
    match p
        case #just (x, y) -> (x, y)
        case #nothing -> (-1, 0)

def step1m [m] (f: [m]i64) (n: i64) (i: i64): (i64, i64) =
    (f[i], f[f[n]])

def step2m [m] (f: [m]i64) (n: i64) (i: i64): (i64, i64) =
    (i, f[f[n]])

def step3m [m] (f: [m]i64) (_: i64) (i: i64): (i64, i64) =
    (i, f[f[i]])

def step [m] (g : [m]i64 -> i64 -> i64 -> (i64, i64))
    (ns: [m][8]maybe_idx) (f: [m]i64) (fn: *[m]i64): *[m]i64 =
    let s = 8 * m
    let (idx, itm) = f |> indices >->
        map (\x -> map
            (\y -> match y
                case #just i -> #just (g f i x)
                case #nothing -> #nothing
            ) ns[x]
        ) >->
        flatten_to s >-> map from_maybe_pair >-> unzip in
    reduce_by_index fn i64.min i64.highest idx itm

def fast_sv_step [m] (ns: [m][8]maybe_idx) ((f, _): ([m]i64, bool)): *([m]i64, bool) =
    let f1 = (step step1m) ns f (copy f)
    let f2 = (step step2m) ns f f1
    let f3 = (step step3m) ns f f2 in
    (f3, any (uncurry (!=)) <-< zip f <| f3)

def is_neighbour (a: cell) (b: cell) =
    let p0 = a.position.0 - b.position.0
    let p1 = a.position.1 - b.position.1 in
    a.event == b.event && a.geometry == b.geometry && p0 * p0 <= 1 && p1 * p1 <= 1

def get_neighbours1 [m] (n: [m]cell) (c: cell): *[8]maybe_idx =
    let ids = (n |> zip (indices n) >-> filter ((.1) >-> is_neighbour c) |> map (\x -> #just x.0)) in
    spread 8 #nothing (indices ids) ids

def get_neighbours [m] (n: [m]cell): *[m][8]maybe_idx =
    map (get_neighbours1 n) n

def get_parent_vector [m] (n: [m]cell): *[m]i64 =
    let ns = get_neighbours n in
    (.0) <| iterate_while (.1) (fast_sv_step ns) (indices n, true)

def get_parents [m] (n: [m]i64): *[]i64 =
    let q = filter (uncurry (==)) (zip n (indices n)) in map (.0) q

def (<+>) ((a1, a2, a3): (f32, f32, f32)) ((b1, b2, b3): (f32, f32, f32)) = (a1 + b1, a2 + b2, a3 + b3)

def red_op (n_a, mx_a, mx2_a, my_a, my2_a) (n_b, mx_b, mx2_b, my_b, my2_b): (f32, f32, f32, f32, f32) =
    let n_ab = n_a + n_b in
    if (n_ab == 0.0) then
        (0.0, 0.0, 0.0, 0.0, 0.0)
    else
        let f_a = n_a / n_ab
        let f_b = n_b / n_ab
        let f_ab = (n_a * n_b) / n_ab
        let deltax = mx_b - mx_a
        let deltay = my_b - my_a
        in (n_ab,
        f_a * mx_a + f_b * mx_b, mx2_a + mx2_b + deltax * deltax * f_ab,
        f_a * my_a + f_b * my_b, my2_a + my2_b + deltay * deltay * f_ab)

def make_measurement [n] (cs: [n]cell): measurement =
    let (a, mx, mx2, my, my2) = reduce_comm red_op (0.0, 0.0, 0.0, 0.0, 0.0)
        (map (\a -> (a.activation, f32.i64 a.position.0, 0.0, f32.i64 a.position.1, 0.0)) cs) in
    {event = (.event) <| (head cs),
    geometry = (.geometry) <| (head cs),
    position = (mx, my),
    variance = (mx2 / a, my2 / a)}

-- TODO: Rewrite using "hist"
def get_measurement [n] (cs: [n]cell) (f: [n]i64) (i: i64): measurement =
    let ps = filter ((== i) <-< (.0)) (zip f (indices f))
    let v = gather cs <-< map (.1) <| ps in
    make_measurement v

def cells_to_measurements [m] (n: [m]cell): *[]measurement =
    let f = get_parent_vector n in
    map (get_measurement n f) (get_parents f)

def unzip6 [n] 'a 'b 'c 'd 'e 'f (xs: [n](a, b, c, d, e, f)): ([n]a, [n]b, [n]c, [n]d, [n]e, [n]f) =
  let (as, bs, cs, ds, des) = unzip5 (map (\(a, b, c, d, e, f) -> (a, b, c, d, (e, f))) xs)
  let (es, fs) = unzip des
  in (as, bs, cs, ds, es, fs)

entry cells_to_measurements_entry [m] (es: [m]u64) (gs: [m]u64) (c0s: [m]i64) (c1s: [m]i64) (as: [m]f32): ([]u64, []u64, []f32, []f32, []f32, []f32) =
    (zip5 es gs c0s c1s as) |>
    map (\(e, g, c0, c1, a) -> {event=e, geometry=g, position=(c0, c1), activation=a}) >->
    cells_to_measurements |>
    map (\x -> (x.event, x.geometry, x.position.0, x.position.1, x.variance.0, x.variance.1)) >->
    unzip6
