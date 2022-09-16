-- TRACCC library, part of the ACTS project (R&D line)
--
-- (c) 2022 CERN for the benefit of the ACTS project
--
-- Mozilla Public License Version 2.0

-- The implementation corresponds to Algorithm 2 of the following paper:
-- https://epubs.siam.org/doi/pdf/10.1137/1.9781611976137.5

import "edm"
import "maybe"

-- This function describes the first step (stochastic hooking) of the FastSV
-- algorithm, and is to be use as an argument for the higher-order `step`
-- function. Takes an existing parent vector f, the index of the current thread
-- i, and the index of a neighbour n.
def step1m [m] (f: [m]i64) (n: i64) (i: i64): (i64, i64) = (f[i], f[f[n]])

-- The second step, aggressive hooking, of the FastSV algorithm. Takes the same
-- arguments as `step1m`.
def step2m [m] (f: [m]i64) (n: i64) (i: i64): (i64, i64) = (i, f[f[n]])

-- The third and final step, shortcutting, in FastSV. Has the same semantics as
-- the other two steps.
def step3m [m] (f: [m]i64) (_: i64) (i: i64): (i64, i64) = (i, f[f[i]])

-- Higher order stepping function for the FastSV algorithm. Given a step
-- function `g` (either `step1m`, `step2m`, or `step3m`), up to eight possible
-- neighbours `ns`, and an old and new parent vector `f` and `fn`, produce a
-- new parent vector using the step function. In other words, `step step1m` is
-- a function that represents stochastic hooking, etc.
def step [m] (g : [m]i64 -> i64 -> i64 -> (i64, i64))
    (ns: [m][8](Maybe i64)) (f: [m]i64) (fn: *[m]i64): *[m]i64 =
    -- s depicts the final size of the neighbourhood array, which is eight
    -- times the number of nodes.
    let s = 8 * m
    -- Gather two lists of indices, where each pair indicates that the second
    -- index is a potential parent for the first.
    let (idx, itm) = f |> indices >->
        map (\x -> map (fmap (\i -> g f i x)) ns[x]) >->
        flatten_to s >-> map (maybe (-1, 0)) >-> unzip in
    -- Then, reduce the indices; since we are looking for the lowest index and
    -- the min operation is commutative, we can easily reduce in this fashion,
    -- using the max i64 value as the neutral element.
    reduce_by_index fn i64.min i64.highest idx itm

-- A single meta-step of the FastSV algorithm, encompassing the three sub-steps
-- described above. Takes a neighbourhood matrix for all nodes and returns a
-- new parent vector, as well as information about whether the vector changed
-- (and, therefore, if another step is necessary).
def fast_sv_step [m] (ns: [m][8](Maybe i64)) ((f, _): ([m]i64, bool)):
    *([m]i64, bool) =
    -- Simply pass the same parent vector through the three distinct steps of
    -- the FastSV algorithm.
    let f1 = (step step1m) ns f (copy f)
    let f2 = (step step2m) ns f f1
    let f3 = (step step3m) ns f f2 in
    -- Return the result, but also check whether the parent vector has changed
    -- at all.
    -- TODO: This check could be integrated into the `step` function to make it
    -- more efficient.
    (f3, any (uncurry (!=)) <-< zip f <| f3)

-- Check whether two cells can possibly be related. This is not the same as
-- a neighbourhood check, but in an ordered list of cells, once this is false,
-- all following cells cannot possibly be neighbours.
def is_candidate (a: Cell) (b: Cell) =
    b.position.1 + 1 >= a.position.1 && b.position.1 <= a.position.1 + 1 &&
    b.event == a.event && b.geometry == a.geometry

-- The actual neighbour check. Returns true iff the two cells are adjacent (by
-- 8-connectivity).
def is_neighbour (a: Cell) (b: Cell): bool =
    let p0 = a.position.0 - b.position.0
    let p1 = a.position.1 - b.position.1 in
    (!(a.position.0 == b.position.0 && a.position.1 == b.position.1)) &&
    b.event == a.event && b.geometry == a.geometry &&
    p0 * p0 <= 1 && p1 * p1 <= 1

-- Helper function that gathers up to 4 neighbours per node, iterating over
-- nodes using some iteration function `f` (which is either increment or
-- decrement)!
def get_neighbours1_helper [m] (n: [m]Cell) (f: i64 -> i64) (i: i64):
    [4](Maybe i64) =
    -- Starting with the current index, iterate one (to move away from the
    -- start), and then keep collecting elements until we have 4 or until there
    -- are no more valid candidates.
    let (vn, _, _) = iterate_while
        -- TODO: It is unclear why the `k < 4` check is necessary.
        (\(_, k, j) -> j >= 0 && j < m && is_candidate n[i] n[j] && k < 4)
        (\(v, k, j) -> if is_neighbour n[i] n[j]
            then ((copy v) with [k] = #Just j, k + 1, f j)
            else (v, k, f j)
        ) (replicate 4 #Nothing, 0, f i) in vn

-- Get the neighbours for a given node by traversing nodes in increasing and
-- decreasing order of index. Since we can find up to 4 elements per direction,
-- we find up to 8 neighbours in total.
def get_neighbours1 [m] (n: [m]Cell) (i: i64): [8](Maybe i64) =
    ((get_neighbours1_helper n (\x -> x - 1) i) ++
     (get_neighbours1_helper n (\x -> x + 1) i)) :> [8](Maybe i64)

-- Commutative reduction operation for Welford's online algorithm for
-- two-dimensional values.
def red_op
    (e_a, g_a, n_a, m_a: (f32, f32), m2_a: (f32, f32))
    (e_b, g_b, n_b, m_b: (f32, f32), m2_b: (f32, f32)):
    (u64, u64, f32, (f32, f32), (f32, f32)) =
    let n_ab = n_a + n_b in
    if (n_ab == 0.0) then
        (0, 0, 0.0, (0.0, 0.0), (0.0, 0.0))
    else
        let delta = (m_b.0 - m_a.0, m_b.1 - m_a.1)
        in (u64.max e_a e_b, u64.max g_a g_b, n_ab, (
                (n_a / n_ab) * m_a.0 + (n_b / n_ab) * m_b.0,
                (n_a / n_ab) * m_a.1 + (n_b / n_ab) * m_b.1
            ), (
                m2_a.0 + m2_b.0 + delta.0 * delta.0 * ((n_a * n_b) / n_ab),
                m2_a.1 + m2_b.1 + delta.1 * delta.1 * ((n_a * n_b) / n_ab)
            )
        )

-- Finally, the actual algorithm. Takes a list of cells, performs FastSV using
-- a neighbour-based image-to-graph conversion, and then returns a list of
-- measurements.
def cells_to_measurements_impl [m] (n: [m]Cell): *[]Measurement =
    -- First, get the 8*m neighbours for each of the nodes.
    let ns = map (get_neighbours1 n) (indices n)
    -- Then, produce the parent vector through repeated application of FastSV.
    let f = (.0) <| iterate_while (.1) (fast_sv_step ns) (indices n, true)
    -- Convert the cells into tuples compatible with Welford's algorithm.
    let i = map (\x -> (x.event, x.geometry, x.activation,
                        (f32.i64 x.position.0, f32.i64 x.position.1),
                        (0.0, 0.0))) n
    -- Commutatively reduce the cells such that we obtain an array of
    -- measurements, but such that only the measurement at parent positions are
    -- valid.
    let h = hist red_op (0, 0, 0.0, (0.0, 0.0), (0.0, 0.0)) m f i |>
        map (\(e, g, w, p, v) -> {
        event = e,
        geometry = g,
        position = (p.0, p.1),
        variance = (v.0 / w, v.1 / w)}
    ) in
    -- Perform a gather operation, grabbing the measurements at those positions
    -- which are parents.
    indices f |> filter (\i -> i == f[i]) |> map (\i -> h[i])
