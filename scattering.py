import numpy as np
from numpy.random import default_rng
from more_itertools import chunked
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

def check_if_transmitted(scattering_result):
    "checks which particles maintained their direction after collision"
    momenta = scattering_result[:,[1,3]]
    return momenta[:,0] * momenta[:,1] > 0 


def transmission_coeff(scattering_result):
    tr = check_if_transmitted(scattering_result)
    return np.sum(tr)/tr.size


def average_p_and_transmission_coeff(scattering_result):
    p_mean = np.mean(scattering_result[:,1])
    tr_coeff = transmission_coeff(scattering_result)
    return p_mean, tr_coeff


def sort_scattering_result(scattering_result):
    return scattering_result[scattering_result[:,1].argsort()]

def theoretical_transmission_coeff_formula(p_i, wp):
    return wp.vp * 2 * np.sqrt(2) *np.pi/(p_i-wp.vp)

def theoretical_transmission_coeff_i(p_i, wp):
    vp = wp.vp
    A = wp.A
    if p_i <= -np.sqrt(2 * A) or p_i >= np.sqrt(2*A):
        return 1.0

    if p_i <= 0.0:
        return 0
    
    out = theoretical_transmission_coeff_formula(p_i, wp)

    out = min(1, out)
    out = max(0, out)

    return out


def theoretical_transmission_coeff_corrected_i(p_i, wp):
    vp = wp.vp
    A = wp.A
    if p_i <= -np.sqrt(2 * A) or p_i >= np.sqrt(2*A):
        return 1.0

    if p_i <= 0.0:
        return 0
    
    out = theoretical_transmission_coeff_formula(p_i, wp)/(2*np.pi)

    out = min(1, out)
    out = max(0, out)

    return out

def theoretical_transmission_coeff(p, wp):
    return np.array([theoretical_transmission_coeff_i(pi, wp) 
                     for pi in p])


def theoretical_transmission_coeff_corrected(p, wp):
    return np.array([theoretical_transmission_coeff_corrected_i(pi, wp) 
                     for pi in p])

if __name__ == "__main__":
    wp = WavePacket(A=1e-2, sigma=40, k=1, vp=0.015)

    icm = RandomInitialConditionMaker(wp)
    init_points, init_times = icm.make_initial_conditions(num=10000, p_max=5e-1)

    scatterrer = wp.make_integrator(atol=1e-10, rtol=1e-10)

    scat = np.row_stack([scatterrer.integrate(point,t_integr=(t, t+1000000))
                            for point, t in zip(init_points, init_times)])

    out=sort_scattering_result(np.column_stack((init_points,scat)))

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
 

    fig, ax = plt.subplots()
    chunks = 100
    p_av=[]
    tr_c=[]
    for sc_r in chunked(out,out.shape[0]//chunks):
        sc_r=np.row_stack(sc_r)
        p_av_i, tr_c_i = average_p_and_transmission_coeff(sc_r)
        p_av.append(p_av_i)
        tr_c.append(tr_c_i)
    p_av = np.array(p_av)
    tr_c = np.array(tr_c)

    ax.plot(p_av, tr_c)

    tr_theoretical = theoretical_transmission_coeff_corrected(p_av, wp)

    ax.plot(p_av, tr_theoretical,'r--')


    plt.show()

    

