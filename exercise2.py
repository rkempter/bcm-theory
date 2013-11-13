import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

rounds = 30000
ys = ([], [], [], [], [])


class postsynaptic_neuron:

	def __init__(self):
		self.omega = np.random.normal(3.0, 1.0, 100)
		self.omega[self.omega < 0] = 0
		self.lrate = 0.05
		self.theta = 100
		self.delta = 2.5
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


means = [10, 30, 50, 70, 90]
omega = 10
input_neurons = np.arange(1,101)

postneuron = postsynaptic_neuron()

for i in range(0, rounds):
	i = np.random.randint(0,5,1)
	presynaptic = norm.pdf(input_neurons, means[i], omega)
	om = postneuron.bcm_rule(presynaptic)
	# compute y for each of the five gaussians
	for j in range(0, 5):
		ys[j].append(np.inner(om, norm.pdf(input_neurons, means[j], omega)))

[f_omega, f_deltas, f_y] = postneuron.get_parameters()

# compute the skewness for each iteration
sum = float(0);
skewness = []
for index, val in enumerate(f_y):
	sum += val
	skewness.append(sum / (index+1))

print skewness

fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.set_title('Weights evolution')
ax2.set_title('Delta evolution')
ax3.set_title('Y')
ax4.set_title('Skewness')

ax1.set_xlim(0, 100)
ax2.set_xlim(0, rounds)
ax3.set_xlim(0, rounds)
ax4.set_xlim(0, rounds)

ax1.plot(f_omega)
ax2.plot(f_deltas)
ax4.plot(skewness)

for i in range(0, 5):
	ax3.plot(ys[i])

