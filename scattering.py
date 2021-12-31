import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
#  from wavepacket import PyWavePacket as WavePacket
from wavepacket import WavePacket

rng = default_rng()

def random(start, stop, num=50):
    width = stop - start
    return start + width * rng.random(num)


class RandomInitialConditionMaker():
    """
    Makes random initial conditions uniformly distributed in momentum and time.

    """
    
    def __init__(self, wp, p_eps=1e-4):
        self.wp=wp
        self.x_wall = 5*wp.sigma
        self.p_eps = p_eps

    def _make_initial_points(self, num, p_range, x_wall):
        p0 = random(*p_range, num=num)
        x0 = np.full_like(p0, fill_value=x_wall)
        return np.column_stack((x0.ravel(), p0.ravel()))

    def make_initial_points(self, num, p_max):
        pos = self._make_initial_points(
            num,
            p_range=(self.p_eps, p_max),
            x_wall=-self.x_wall,
        )
        neg = self._make_initial_points(
            num,
            p_range=(-p_max, -self.p_eps),
            x_wall=self.x_wall,
        )

        return np.row_stack((pos,neg))

    def make_initial_times(self, num):
        omega = self.wp.k * self.wp.vp
        T = 2*np.pi/omega
        return random(0,T,num)
    
    def make_initial_conditions(self, num, p_max):
        points = self.make_initial_points(num, p_max)
        times = self.make_initial_times(2*num)

        return points, times

if __name__ == "__main__":
    wp = WavePacket(A=1e-2, sigma=40, k=1, vp=0.01)

    icm = RandomInitialConditionMaker(wp)
    init_points, init_times = icm.make_initial_conditions(num=10000, p_max=5e-1)

    scatterrer = wp.make_integrator(atol=1e-10, rtol=1e-10)

    scat = np.row_stack([scatterrer.integrate(point,t_integr=(t, t+1000000))
                            for point, t in zip(init_points, init_times)])

    out=np.column_stack((init_points,scat))

    fig, ax = plt.subplots()
    ax.plot(out[:,1], out[:,-1]-out[:,1],',k',alpha=0.2)
    ax.set_aspect("equal")
    ax.axvspan(xmin=np.sqrt(2 * wp.A)+wp.vp,
             xmax=max(ax.get_xlim()),
             alpha=0.2,
    )
    ax.axvspan(xmin=min(ax.get_xlim()),
             xmax=-np.sqrt(2 * wp.A)+wp.vp,
             alpha=0.2,
    )
    ax.axvspan(xmin=-np.sqrt(2 * wp.A)+wp.vp,
             xmax=np.sqrt(2 * wp.A)+wp.vp,
             alpha=0.2,
             color='r')

    fig, ax = plt.subplots()
    ax.plot(out[:,1], out[:,-1]-out[:,1],',k',alpha=1)
    ax.set_aspect("equal")
    ax.axvspan(xmin=np.sqrt(2 * wp.A)+wp.vp,
             xmax=max(ax.get_xlim()),
             alpha=0.2,
    )
    ax.axvspan(xmin=min(ax.get_xlim()),
             xmax=-np.sqrt(2 * wp.A)+wp.vp,
             alpha=0.2,
    )
    ax.axvspan(xmin=-np.sqrt(2 * wp.A)+wp.vp,
             xmax=np.sqrt(2 * wp.A)+wp.vp,
             alpha=0.2,
             color='r')
 
    plt.show()

