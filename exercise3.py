import numpy as np
import matplotlib.pyplot as plt
import Image

class receptiveField:

	rounds = 150000

	def __init__(self):
		self.lrate = 5 * 10 ** (-6)
		self.theta = 1200
		self.delta = 5

		self.omega = np.random.normal(0.5, 0.15, 512)
		self.omega[self.omega < 0] = 0

	# load images from folder
	def loadImages(self):
		# put some exception handling here
		images = []

		for i in range(1, 11):
			im = Image.open('images/im{0}.bmp'.format(i))
			images.append(np.array(im))

		return np.array(images, dtype='float')

	# normalize (mean = 0, std = 1)
	def normalization(self, images):
		for image in images:
			mean = np.mean(image)
			std = np.std(image)
			image -= mean
			image /= std

		return images

	# create the input vectors (prepare for + and -)
	def createInputVectors(self, images):
		# extract 5000 patches out of each image
		patches = np.empty([50000, 512])

		for index, image in enumerate(images):

			# create patches
			i = 0; j = 0
			# create 5000 patches (difference between succeeding patches x: 4, y: 14)
			for k in range(0, 5000):
				# move in grid
				if i >= 486:
					i = 0
					j += 14

				patch = image[i:(i+16),j:(j+16)].flatten()

				patches[index*5000+k,0:256] = (patch > 0) * patch
				patches[index*5000+k,256:512] = (patch < 0) * np.abs(patch)
				
				# move in grid
				i += 5

		return patches

	# Hebbian Learning rule implementation (BCM)
	def bcmTraining(self, input):
		y = np.inner(self.omega, input)
		d_omega = self.lrate * input * (y ** 2 - y * self.delta)
		self.omega += d_omega
		self.omega[self.omega < 0] = 0
		d_delta = (float(1) / self.theta * (y ** 2 - self.delta))
		self.delta += d_delta

	# Plot result in a pcolor figure
	def plotResults(self, omega):
		rcpField = []

		for index, om in enumerate(omega):
			rcpField.append(np.array(om[0:256] - om[256:512]));

		fig = plt.figure()

		fig.suptitle('Receptive field after 5k, 70k, 110k and 150k iterations', fontsize=14, fontweight='bold')
		fig.subplots_adjust(top=0.85, wspace=0.3, hspace=0.4, right=0.8)

		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)

		im1 = ax1.imshow(rcpField[0].reshape([16,16]), cmap='RdBu')
		im2 = ax2.imshow(rcpField[1].reshape([16,16]), cmap='RdBu')
		im3 = ax3.imshow(rcpField[2].reshape([16,16]), cmap='RdBu')
		im4 = ax4.imshow(rcpField[3].reshape([16,16]), cmap='RdBu')

		ax1.axis([0, 15, 0, 15])
		ax2.axis([0, 15, 0, 15])
		ax3.axis([0, 15, 0, 15])
		ax4.axis([0, 15, 0, 15])
		
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		plt.colorbar(im4, cax = cbar_ax)

		ax1.set_title('30k iter')
		ax2.set_title('70k iter')
		ax3.set_title('110k iter')
		ax4.set_title('150k iter')

		fig.show()

	# Main: load, normalize, learn, plot
	def main(self):
		images = self.loadImages()
		normalizedImages = self.normalization(images)
		inputVectors = self.createInputVectors(normalizedImages)

		intermediate_omegas = []

		for i in range(0, receptiveField.rounds):
			vectorNbr = np.random.randint(0, 50000, 1)
			self.bcmTraining(inputVectors[vectorNbr,:].flatten())
			if i == 30000 or i == 70000 or i == 110000:
				intermediate_omegas.append(np.copy(self.omega))

		intermediate_omegas.append(self.omega)

		print np.size(intermediate_omegas)

		self.plotResults(intermediate_omegas)


# Create receptive field and train it
rcpFieldInstance = receptiveField()
rcpFieldInstance.main()


