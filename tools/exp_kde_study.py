
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

print("\nBegin kernel density estimation demo ")

np.random.seed(0)

x_data = np.array([0.63222682, 0.63222688, 0.63222694, 0.6321367 , 0.66900623,
       0.64073235, 0.63220823, 0.77164757, 0.77729195, 0.83061945,
       0.76067114, 0.77404934, 0.79521912, 0.82125366, 0.78140873,
       0.81886661, 0.83973408, 0.90340763, 0.78140867, 0.81886655,
       0.83973402, 0.90340751])
x_data = np.sort(x_data)
print("\nSource data points (normal): ")
weights = np.ones(len(x_data))
weights[int(len(x_data)/2):] = 0
# x_data = x_data[weights > 0]
# weights = weights[weights > 0]

print(list(zip(x_data,weights)))
print("\nGenerating estimated PDF function from source x_data ")
kde = stats.gaussian_kde(x_data, weights=weights)

x = np.linspace(min(x_data[weights > 0]), max(x_data[weights > 0]),100)
estimated_pdf = kde(x)
# y_normal = stats.norm.pdf(x_pts)

plt.figure()
# plt.hist(x_data, bins=7, density=1.0)
plt.plot(x, estimated_pdf, label="kde estimated PDF", \
 color="r")
plt.legend()
plt.show()
