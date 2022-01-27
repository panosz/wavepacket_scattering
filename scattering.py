import numpy as np
from numpy.random import default_rng
from more_itertools import chunked
import matplotlib.pyplot as plt
from interval import interval, inf
#  from wavepacket import PyWavePacket as WavePacket
from wavepacket import WavePacket

rng = default_rng(seed=42)


def random(start, stop, num=50):
    width = stop - start
    return start + width * rng.random(num)


class RandomInitialConditionMaker():
    """
    Makes random initial conditions uniformly distributed in momentum and time.

    """

    def __init__(self, wp, p_eps=1e-4):
        self.wp = wp
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

        return np.row_stack((pos, neg))

    def make_initial_times(self, num):
        omega = self.wp.k * self.wp.vp
        T = 2*np.pi/omega
        return random(0, T, num)

    def make_initial_conditions(self, num, p_max):
        points = self.make_initial_points(num, p_max)
        times = self.make_initial_times(2*num)

        return points, times


class ScatteringResult(np.ndarray):
    """
    Array containing the output of the scattering of a particle distribution
    from a wavepacket.
    """
    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def calculate_delta_p(self):
        momenta = self[:, [1, 3]]
        return momenta[:, 1] - momenta[:, 0]

    def check_if_transmitted(self):
        "checks which particles maintained their direction after collision"
        momenta = self[:, [1, 3]]
        return momenta[:, 0] * momenta[:, 1] > 0

    def transmission_coeff(self):
        tr = self.check_if_transmitted()
        return np.sum(tr)/tr.size

    def average_p_and_transmission_coeff(self, v_f=None):
        p_mean = np.mean(self[:, 1])

        if v_f is None:
            tr_coeff = self.transmission_coeff()
        else:
            scattering_result_in_frame = self.change_frame_of_ref(
                v_f,
            )
            tr_coeff = scattering_result_in_frame.transmission_coeff()
        return p_mean, tr_coeff

    def sort_wrt_p_init(self):
        return self[self[:, 1].argsort()].copy()

    def change_frame_of_ref(self, v_f):
        out = self.copy()
        out[:, [1, 3]] = out[:, [1, 3]] - v_f
        return out

    def chunks(self, num):
        """
        iterate in chunks
        """
        for sc_r in chunked(self,
                            self.shape[0]//num):
            yield type(self)(np.row_stack(sc_r))


class Transmission_coeff_calculator_Base():
    def __init__(self, wp):
        self.wp = wp

    @staticmethod
    def clip(x):
        return np.clip(x, a_min=0, a_max=1)

    def transmission_coeff_formula_in_ref_frame(self, p_i):
        return self.wp.vp * np.sqrt(2) / p_i

    def theoretical_transmission_coeff_formula(self, p_i):
        return self.transmission_coeff_formula_in_ref_frame(p_i - self.v_ref)

    def __call__(self, p):
        return np.array([self.theoretical_transmission_coeff_i(pi)
                         for pi in p])

    def theoretical_transmission_coeff_i(self, p_i):
        if p_i not in self.interaction_interval:
            return 1.0

        if p_i in self.total_reflection_interval:
            return 0

        if p_i in self.total_transmission_interval:
            return 1.0

        out = self.theoretical_transmission_coeff_formula(p_i)

        return self.clip(np.abs(out))


class HeuristicTransmissionCoefCalculator(Transmission_coeff_calculator_Base):

    @property
    def v_ref(self):
        return wp.vp / 2

    @property
    def interaction_interval(self):
        A = wp.A
        vp = self.wp.vp
        p_crit = np.sqrt(2 * A)
        return interval[-p_crit + 3 * vp/4,
                        p_crit + vp]

    @property
    def total_transmission_interval(self):
        return interval[self.wp.vp/2, self.wp.vp]

    @property
    def total_reflection_interval(self):
        return interval([-inf, self.wp.vp/2]) & self.interaction_interval


class TransmissionCoefCalculator(Transmission_coeff_calculator_Base):

    @property
    def v_ref(self):
        return wp.vp

    @property
    def interaction_interval(self):
        A = wp.A
        vp = self.wp.vp
        p_crit = np.sqrt(2 * A)
        return interval[-p_crit + vp,
                        p_crit + vp]

    @property
    def total_transmission_interval(self):
        return interval[0, self.wp.vp]

    @property
    def total_reflection_interval(self):
        return interval([-inf, 0]) & self.interaction_interval


if __name__ == "__main__":
    wp = WavePacket(A=1e-2, sigma=40, k=2, vp=0.045)

    icm = RandomInitialConditionMaker(wp)
    init_points, init_times = icm.make_initial_conditions(num=10000,
                                                          p_max=5e-1)

    scatterrer = wp.make_integrator(atol=1e-10, rtol=1e-10)

    scat = np.row_stack([scatterrer.integrate(point, t_integr=(t, t+1000000))
                         for point, t in zip(init_points, init_times)])

    out = ScatteringResult(np.column_stack((init_points, scat)))

    out = out.sort_wrt_p_init()

    fig, ax = plt.subplots()
    ax.plot(out[:, 1], out[:, -1]-out[:, 1], ',k', alpha=0.2)
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
    ax.axvline(wp.vp)

    fig, ax = plt.subplots()
    ax.plot(out[:, 1]/wp.vp, (out[:, -1]-out[:, 1])/wp.vp, ',k', alpha=0.2)
    ax.set_aspect("equal")
    ax.axvspan(xmin=(np.sqrt(2 * wp.A)+wp.vp)/wp.vp,
               xmax=max(ax.get_xlim()),
               alpha=0.2,
               )
    ax.axvspan(xmin=min(ax.get_xlim()),
               xmax=(-np.sqrt(2 * wp.A)+wp.vp)/wp.vp,
               alpha=0.2,
               )
    ax.axvspan(xmin=(-np.sqrt(2 * wp.A)+wp.vp)/wp.vp,
               xmax=(np.sqrt(2 * wp.A)+wp.vp)/wp.vp,
               alpha=0.2,
               color='r')

    fig, ax = plt.subplots()
    chunks = 1000
    p_av = []
    tr_c = []
    for sc_r in out.chunks(chunks):
        p_av_i, tr_c_i = sc_r.average_p_and_transmission_coeff(v_f=wp.vp/2)
        p_av.append(p_av_i)
        tr_c.append(tr_c_i)
    p_av = np.array(p_av)
    tr_c = np.array(tr_c)

    ax.plot(p_av, tr_c)

    tr_heuristic = HeuristicTransmissionCoefCalculator(wp)(p_av)
    tr_theoretical = TransmissionCoefCalculator(wp)(p_av)

    ax.plot(p_av, tr_theoretical, 'r--', alpha=0.7)
    #  ax.plot(p_av, tr_heuristic, 'k--', alpha=0.7)

    delta_p = out.calculate_delta_p()

    delta_p_av = np.mean(delta_p)

    delta_p_av_pos_init = np.mean(delta_p[out[:, 1] > 0])
    delta_p_av_neg_init = np.mean(delta_p[out[:, 1] < 0])

    print(f"average delta_p = {delta_p_av}")
    print(f"average delta_p for positive initial p = {delta_p_av_pos_init}")
    print(f"average delta_p for negative initial p = {delta_p_av_neg_init}")

    plt.show()
