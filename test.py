import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

dx, dy = 0.05, 0.05

y, x = np.mgrid[slice(1, 5+dy, dy), slice(1, 5+dx, dx)]
z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
z = z[:-1,:-1]
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

cmap = plt.get_cmap("PiYG")
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, (ax0, ax1) = plt.subplots(nrows=2)

im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
ax0.set_title("pcolormesh with levels")

cf = ax1.contourf(x[:-1, :-1] + dx/2, y[:-1,:-1] + dy/2, z, levels=levels, cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title("contourf with levels")

fig.tight_layout()
plt.show()
