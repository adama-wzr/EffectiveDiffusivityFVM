import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# simulation properties

xSize = 1002
ySize = 2007

tol = 1e-9

# Read image
imgName = "00042.jpg"
img = np.uint8(mpimg.imread(imgName))

# df1 = pd.read_csv("ExpUV_mod.csv")
# df1 = pd.read_csv("UV.csv")
df1 = pd.read_csv("CMAP_00042.csv")
# get raw data from csv files

raw_data = df1[["X","Y","C"]].to_numpy()

C = raw_data[:,2]
C = np.reshape(C, [ySize, xSize])


# Post-Process C, add a mask where C = 0 (solid)

(m,n) = C.shape

mask = np.zeros_like(C, dtype=bool)

for i in range(m):
	for j in range(n):
		if C[i][j] < tol:
			mask[i][j] = True		



C = np.ma.array(C, mask=mask)

# Create the mesh grid

Xp, Yp = np.meshgrid(np.linspace(0, xSize, xSize), np.linspace(ySize, 0, ySize))

# plotting

fig1, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

fig1.set_dpi(100)
fig1.set_size_inches(8, 7)

# First axis is just the image

ax1.imshow(img)
ax1.set_title(imgName)
ax1.axes.set_aspect('equal')

# Second axis is Concentration contour

CS2 = ax2.contourf(Xp, Yp, C, 40, cmap=plt.cm.inferno)
cbar2 = fig1.colorbar(CS2, ax=ax2)
ax2.set_title("Concentration Contour")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.axes.set_aspect('equal')

plt.show()
