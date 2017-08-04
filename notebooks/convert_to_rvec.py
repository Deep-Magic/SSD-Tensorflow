import numpy as np

x, y, z = -23, 45, -10
x = np.deg2rad(x)
y = np.deg2rad(y)
z = np.deg2rad(z)

x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)],[0, np.sin(x), np.cos(x)]])
y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

rvec = np.matmul(np.matmul(z,y),x)

print(rvec)
