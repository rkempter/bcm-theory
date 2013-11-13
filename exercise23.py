import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

rounds = 10000
ys = ([], [])


class postsynaptic_neuron:

	def __init__(self):
		self.omega = np.array([float(1),float(1)])
		self.lrate = 0.005
		self.theta = 100
		self.delta = 2
		self.realY = []

		self.deltas = []

	def bcm_rule(self, input):
		y = np.inner(self.omega, input)
		self.realY.append(y)
		d_omega = self.lrate * input * (y**2 - y * self.delta)
		self.omega += d_omega
		self.omega[self.omega < 0] = 0
		d_delta = (float(1) / self.theta * (y**2 - self.delta))
		self.delta += d_delta
		self.deltas.append(self.delta)

		return self.omega

	def get_parameters(self):
		return [self.omega, self.deltas, self.realY]


input = [np.array([1,0]), np.array([0,1])]

postneuron = postsynaptic_neuron()

omega1 = []
omega2 = []

for i in range(0, rounds):
	i = np.random.randint(0,2,1)
	om = postneuron.bcm_rule(input[i])
	print om[1]
	omega1.append(om[0])
	omega2.append(om[1])
	# compute y for each of the five gaussians

[f_omega, f_deltas, f_y] = postneuron.get_parameters()

# compute the skewness for each iteration
fig = plt.figure()
fig.suptitle('Simulation for selectivity', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85, hspace=0.6)
ax1 = fig.add_subplot(311)

ax1.set_title('Weights evolution')

ax1.set_xlim(0, rounds)
ax1.plot(omega1)
ax1.plot(omega2)
ax1.plot(f_deltas)

# for i in range(0, 1):
# 	ax2.plot(ys[i])

fig.show()

