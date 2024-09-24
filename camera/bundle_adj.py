fx = 200
fy = 200
x0 = 100
y0 = 100
k1 = 0.2
k2 = 0.2
k3 = 0.2
p1 = 0.2
p2 = 0.2

r11 = 1
r12 = 0
r13 = 0
r21 = 0
r22 = 1
r23 = 0
r31 = 0
r32 = 0
r33 = 1
Xs = 1
Ys = 1
Zs = 1


optv = [fx, fy, x0, y0, k1, k2, k3, p1, p2, r11, r12, r13, r21, r22, r23, r31, r32, r33, Xs, Ys, Zs]

X, Y, Z = [0, 0, 0]

x = x0 - fx * (r31 * (X - Xs) + r32 * (Y - Ys) + r33 * (Z - Zs)) / (r11 * (X - Xs) + r12 * (Y - Ys) + r13 * (Z - Zs))
print(f"> x: {x}")
y = y0 - fy * (r31 * (X - Xs) + r32 * (Y - Ys) + r33 * (Z - Zs)) / (r21 * (X - Xs) + r22 * (Y - Ys) + r23 * (Z - Zs))
print(f"> y: {y}")
