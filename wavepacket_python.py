import numpy as np
import matplotlib.pyplot as plt
#  from wavepacket import PyWavePacket as WavePacket
from wavepacket import WavePacket





if __name__ == "__main__":
    wp = WavePacket(A=1e-2, sigma=10, k=1, vp=0.01)

    p0 = -12e-3
    x0 = np.linspace(50.5,40, num=100000)

    scatterrer = wp.make_integrator()

    scat = np.column_stack([scatterrer.integrate(point=[xi, p0],t_integr=10000)
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

    fig, ax = plt.subplots()
    ax.hist(scat[1,:], bins=1000)


    plt.show()
