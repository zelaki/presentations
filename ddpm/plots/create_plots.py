import numpy as np
import matplotlib.pyplot as plt

b_t = np.linspace(10e-4, 0.02,1000)
a_t = 1 - b_t
a_t_bar = np.cumprod(a_t)
sigma_q_t = b_t

lambda_t = (1 - a_t)**2 / (2 * sigma_q_t**2 * a_t * (1 - a_t_bar))
plt.plot(lambda_t, 'r')
plt.xlabel("step")
plt.ylabel("λ")

plt.savefig('figs/lambda.png', transparent=True)