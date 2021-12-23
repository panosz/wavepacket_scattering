import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Wavepacket():

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


def scatter(wp, point, t_inter):
    sol = solve_ivp(wp.system,[0, t_inter], y0=point,)

    return sol.y[:, -1]



if __name__ == "__main__":
    wp = Wavepacket(A=1e-2, sigma=10, k=1, vp=0.01)

    p0 = -12e-3
    x0 = np.linspace(50.5,40, num=100000)


    scat = np.column_stack([scatter(wp,point=[xi, p0],t_inter=10000)
                            for xi in x0])

    out=np.zeros((4,x0.size))
    out
    out[0,:]=x0
    out[1,:]=p0
    out[2:,:]=scat

    #  sol = solve_ivp(wp.system,[0,10000], y0=[x0,p0],)

    fig, ax = plt.subplots()
    ax.plot(x0,scat[1,:],',k',alpha=0.5)

    fig, ax = plt.subplots()
    ax.plot(scat[0, :],scat[1,:],',k',alpha=0.5)


    plt.show()
