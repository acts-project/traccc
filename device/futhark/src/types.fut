import "linear"

type cell = {
    event: u64,
    geometry: u64,
    position: (i64, i64),
    activation: f32
}

type measurement = {
    event: u64,
    geometry: u64,
    position: (f32, f32),
    variance: (f32, f32)
}

type spacepoint = point3
