import matplotlib.pyplot as plt
import numpy as np


sat_x = 0.5
sat_y = 0.99

fg_scale = 0.164752557
nsa_k = 13.584688855
nsa_y0 = 0.161742724

x = np.arange(0, 0.6, 0.01)

fig = plt.figure(figsize=(7, 2))

plt.plot(x, 1 / (1 + np.exp(-nsa_k * (x - nsa_y0))),
         label=r'NSA: $y = \frac{1}{1+e^{-k(x-x_0)}}$')
plt.plot(x, 1 - np.exp(-0.5 * (x ** 2 / fg_scale ** 2)),
         label='Ours: $y=1 - \\frac{p_X(x)}{p_X(0)}$,\n'
               '$\qquad \quad X\sim \mathcal{N}(0, \sigma^2)$')

plt.xlim(right=0.55)

# Shrink current axis by 20%
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-large')

plt.grid(True)
plt.tight_layout()
plt.show()
fig.savefig('label_fig.pdf')
