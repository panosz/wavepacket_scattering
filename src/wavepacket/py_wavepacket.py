import numpy as np
import matplotlib.pyplot as plt

class PyWavePacket():

    def __init__(self, A, sigma, k, vp):
        self.A = A
        self.sigma_sq = sigma**2
        self.k = k
        self.vp = vp

    def __call__(self, z, t):

        exponent = -z**2/(2*self.sigma_sq)
        phase = self.k*(z - self.vp * t)
        return self.A * np.exp(exponent) * np.sin(phase)

    def dz(self, z, t):
        exponent = -z**2/(2*self.sigma_sq)
        phase = self.k*(z - self.vp * t)
        dz1 = self.A * np.exp(exponent) * self.k * np.cos(phase)
        dz2 = - z/self.sigma_sq * self.A * np.exp(exponent) * np.sin(phase)

        return dz1 + dz2


    def plot(self, ax=None, dz=False):
        if ax is None:
            _, ax = plt.subplots()

        zlim = 4*np.sqrt(self.sigma_sq)
        z = np.linspace(-zlim, zlim, num=2000)

        if dz:
            ax.plot(z, self.dz(z,0))
        else:
            ax.plot(z, self(z, 0))


    def system(self, t, y):
        z, p = y
        dzdt = p
        dpdt = -self.dz(z, t)
        return np.array([dzdt, dpdt])

