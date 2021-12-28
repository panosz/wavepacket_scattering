import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
#  from wavepacket import PyWavePacket as WavePacket
from wavepacket import WavePacket

rng = default_rng()

def random(start, stop, num=50):
    width = stop - start

    return start + width * rng.random(num)
    

def make_initial_points_positive(num):
    p0 = random(1e-4, 5e-1, num=num)
    x0 = random(-50.5,-40, num=num)

    return np.column_stack((x0.ravel(), p0.ravel()))

def make_initial_points_negative(num):
    p0 = random(-1e-4, -5e-1, num=num)
    x0 = random(50.5,40, num=num)

    return np.column_stack((x0.ravel(), p0.ravel()))

def make_initial_points(num):
    pos = make_initial_points_positive(num)
    neg = make_initial_points_negative(num)

    return np.row_stack((pos,neg))


if __name__ == "__main__":
    wp = WavePacket(A=1e-2, sigma=10, k=1, vp=0.01)

    init_points = make_initial_points(10000)

    scatterrer = wp.make_integrator(atol=1e-10, rtol=1e-10)

    scat = np.row_stack([scatterrer.integrate(point,t_integr=(0, 10000))
                            for point in init_points])

    out=np.column_stack((init_points,scat))

    fig, ax = plt.subplots()
    ax.plot(out[:,1], out[:,-1],',k',alpha=0.2)
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

