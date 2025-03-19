import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams["font.family"] = "Times New Roman"

# simulation properties

xSize = 863
ySize = 868

tol = 1e-9

# Read image
imgName = "Cycled_SC811.jpg"
img = np.uint8(mpimg.imread(imgName))

# df1 = pd.read_csv("ExpUV_mod.csv")
# df1 = pd.read_csv("UV.csv")
df1 = pd.read_csv("Cycled_SC811_FMap.csv")
df2 = pd.read_csv("Cycled_SC811_CMap.csv")

# get raw data from csv files
raw_data1 = df1[["x","y","Jx","Jy"]].to_numpy()
raw_data2 = df2[["x", "y", "C"]].to_numpy()

C = np.zeros((ySize,xSize))
Jx = np.zeros((ySize,xSize))
Jy = np.zeros((ySize,xSize))
x = raw_data2[:,0].astype(int)
y = raw_data2[:,1].astype(int)

for i in range(len(raw_data2[:,0])):
	C[y[i]][x[i]] = raw_data2[i,2]
	Jx[y[i]][x[i]] = raw_data1[i,2]
	Jy[y[i]][x[i]] = raw_data1[i,3]

# apply a mask

mask = np.zeros_like(C, dtype=bool)

for i in range(ySize):
	for j in range(xSize):
		if C[i,j] < tol:
			mask[i,j] = True

C = np.ma.array(C, mask=mask)
Jx = np.ma.array(Jx, mask=mask)
Jy = np.ma.array(Jy, mask=mask)

# Create the mesh grid

Xp, Yp = np.meshgrid(np.linspace(0, 1, xSize), np.linspace(1.0*ySize/xSize, 0, ySize))

# plotting

fig1, ((ax1, ax2, ax3)) = plt.subplots(1, 3, constrained_layout=True)

fig1.set_dpi(100)
fig1.set_size_inches(8, 4)

# First axis is just the image

ax1.imshow(img)
ax1.set_title(imgName, fontsize=16)

# Second axis is U-velocity contour

CS2 = ax2.contourf(Xp, Yp, C, 40, cmap=plt.cm.inferno)
cbar2 = fig1.colorbar(CS2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label(r'Concentration Distribution', rotation=90, fontsize=14)
ax2.set_aspect('equal', adjustable='box')

# Third axis is the flux contour

FluxMag = np.sqrt(Jx**2 + Jy**2)

step = 100

# CS3 = ax3.contourf(Xp, Yp, FluxMag, 40, cmap=plt.cm.rainbow)
# # ax3.quiver(Xp[::step, ::step], Yp[::step, ::step], Jx[::step, ::step], Jy[::step,::step])
# cbar3 = fig1.colorbar(CS3, ax=ax3, fraction=0.046, pad=0.04)
# cbar3.set_label('Mass Flux', rotation=90, fontsize=14)
# ax3.set_aspect('equal', adjustable='box')

CS3 = ax3.contourf(Xp, Yp, Jx, 40, cmap=plt.cm.inferno)
cbar3 = fig1.colorbar(CS3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label(r'Flux distribution', rotation=90, fontsize=14)
ax3.set_aspect('equal', adjustable='box')

plt.show()
