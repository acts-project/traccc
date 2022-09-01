type point2 = (f32, f32)

type point3 = (f32, f32, f32)

type affine3 = [4][4]f32

def transform (t: affine3) ((x, y): point2): point3 = (
    x * t[0, 0] + y * t[0, 1] + t[0, 3],
    x * t[1, 0] + y * t[1, 1] + t[1, 3],
    x * t[2, 0] + y * t[2, 1] + t[2, 3]
)
