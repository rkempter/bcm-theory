import scipy as sp
import numpy as np
import pylab as pl

epsilon = np.finfo(np.float64).eps;

def bcm_rule(x0, omega0, delta0, lrate, theta, z):
	omega = omega0;
	delta = delta0;
	x = [0, x0];
	d_omega = 1;

	i = 0;
	while i < 15000:
		i += 1;
		k = np.random.choice(x, p = [1-z, z]);
		y = omega * k;
		d_omega = lrate * k * (y**2 - y * delta);
		omega = omega + d_omega;
		d_delta = (float(1) / theta * (y**2 - delta));
		delta = delta + d_delta;

	return omega * x0;

result = []
X = []

for z in range(2, 9):
	y = bcm_rule(1, 1, 0, .0005, 500, z*0.1);
	X.append(z*0.1);
	result.append(y);

print result;

pl.plot(X, result);