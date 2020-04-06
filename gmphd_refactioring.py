import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
import ospa


def multivariate_gaussian(x: np.ndarray, m: np.ndarray, P: np.ndarray):
    first_part = 1 / (((2 * np.pi) ** (x.size / 2.0)) * (lin.det(P) ** 0.5))
    second_part = -0.5 * (x - m) @ lin.inv(P) @ (x - m)
    return first_part * np.exp(second_part)


def multivariate_gaussian_with_invP(x: np.ndarray, m: np.ndarray, detP, invP: np.ndarray):
    first_part = 1 / (((2 * np.pi) ** (x.size / 2.0)) * (detP ** 0.5))
    second_part = -0.5 * (x - m) @ invP @ (x - m)
    return first_part * np.exp(second_part)


def clutter_intensity_function(z, lc, surveillance_region):
    '''
    Clutter intensity function, with uniform distribution through the surveillance region, pg. 8
    in "Bayesian Multiple Target Filtering Using Random Finite Sets" by Vo, Vo, Clark.
    :param z:
    :param lc: average number of false detections per time step
    :param surveillance_region: np.ndarray of shape (number_dimensions, 2) giving range(min and max) for each dimension
    '''
    if surveillance_region[0][0] <= z[0] <= surveillance_region[0][1] and surveillance_region[1][0] <= z[1] <= \
            surveillance_region[1][1]:
        # example in two dimensions: lc/((xmax - xmin)*(ymax-ymin))
        return lc / ((surveillance_region[0][1] - surveillance_region[0][0]) * (
                surveillance_region[1][1] - surveillance_region[1][0]))
    else:
        return 0


class GaussianMixture:
    def __init__(self, w, m, P):
        self.w = w
        self.m = m
        self.P = P
        self.detP = None
        self.invP = None

    def assign_determinant_and_inverse(self, detP, invP):
        self.detP = detP
        self.invP = invP

    def mixture_value(self, x: np.ndarray):
        sum = 0
        if self.detP is None:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i])
        else:
            for i in range(len(self.w)):
                sum += self.w[i] * multivariate_gaussian_with_invP(x, self.m[i], self.detP[i], self.invP[i])
        return sum

    def mixture_component_value_at(self, x, i):
        if self.detP is None:
            return self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i])
        else:
            return self.w[i] * multivariate_gaussian_with_invP(x, self.m[i], self.detP[i], self.invP[i])

    def mixture_value_split_into_components(self, x):
        val = []
        if self.detP is None:
            for i in range(len(self.w)):
                val.append(self.w[i] * multivariate_gaussian(x, self.m[i], self.P[i]))
        else:
            for i in range(len(self.w)):
                val.append(self.w[i] * multivariate_gaussian_with_invP(x, self.m[i], self.detP[i], self.invP[i]))
        return val

    def copy(self):
        w = self.w.copy()
        m = []
        P = []
        for m1 in self.m:
            m.append(m1.copy())
        for P1 in self.P:
            P.append(P1.copy())
        return GaussianMixture(w, m, P)


class GmphdFilter:
    def __init__(self, model, dtype=np.float64):
        # to do: dtype, copy, improve performance
        self.p_s = model['p_s']
        self.F = model['F']
        self.Q = model['Q']
        self.w_spawn = model['w_spawn']
        self.F_spawn = model['F_spawn']
        self.d_spawn = model['d_spawn']
        self.Q_spawn = model['Q_spawn']
        self.birth_GM = model['birth_GM']
        self.p_d = model['p_d']
        self.H = model['H']
        self.R = model['R']
        self.clutter_density_func = model['clutt_int_fun']
        self.T = model['T']
        self.U = model['U']
        self.Jmax = model['Jmax']

    def thinning_and_displacement(self, v, p, F, Q):
        w = []
        m = []
        P = []
        for w1 in v.w:
            w.append(w1 * p)
        for m1 in v.m:
            m.append(F @ m1)
        for P1 in v.P:
            P.append(Q + F @ P1 @ F.T)
        return GaussianMixture(w, m, P)

    def spawn_mixture(self, v):
        w = []
        m = []
        P = []
        for i in range(len(v.w)):
            for j in range(len(self.w_spawn)):
                w.append(v.w[i] * self.w_spawn[j])
                m.append(self.F_spawn[j] @ v.m[i] + self.d_spawn[j])
                P.append(self.Q_spawn[j] + self.F_spawn[j] @ v.P[i] @ self.F_spawn[j].T)
        return GaussianMixture(w, m, P)

    def getDeterAndInverseof(self, P_list):
        return self.getDeterminantOfList(P_list), self.getInverseOfList(P_list)

    def getDeterminantOfList(self, P_list):
        detP = []
        for P in P_list:
            detP.append(lin.det(P))
        return detP

    def getInverseOfList(self, P_list):
        invP = []
        for P in P_list:
            invP.append(lin.inv(P))
        return invP

    def prediction(self, v):
        # v_pred = v_s + v_spawn +  v_new_born

        # targets that survived v_s:
        v_s = self.thinning_and_displacement(v, self.p_s, self.F, self.Q)
        # spawning targets
        v_spawn = self.spawn_mixture(v)
        # final phd of prediction
        return GaussianMixture(v_s.w + v_spawn.w + self.birth_GM.w, v_s.m + v_spawn.m + self.birth_GM.m,
                               v_s.P + v_spawn.P + self.birth_GM.P)

    def correction(self, v, Z):
        v_residual = self.thinning_and_displacement(v, self.p_d, self.H, self.R)
        detP, invP = self.getDeterAndInverseof(v_residual.P)
        v_residual.assign_determinant_and_inverse(detP, invP)

        K = []
        P_kk = []
        for i in range(len(v_residual.w)):
            k = v.P[i] @ self.H.T @ invP[i]
            K.append(k)
            P_kk.append(v.P[i] - k @ self.H @ v.P[i])

        w = []
        m = []
        P = []
        for z in Z:
            values = v_residual.mixture_value_split_into_components(z)
            normalization_factor = np.sum(values) + self.clutter_density_func(z)
            for i in range(len(v_residual.w)):
                w.append(values[i] / normalization_factor)
                m.append(v.m[i] + K[i] @ (z - self.H @ v.m[i]))
                P.append(P_kk[i])

        w.extend((np.array(v.w) * (1 - self.p_d)).tolist())
        m.extend(v.m)
        P.extend(v.P)

        return GaussianMixture(w, m, P)

    def pruning(self, v: GaussianMixture):
        I = list(range(len(v.w)))
        invP = self.getInverseOfList(v.P)
        while len(I) > 0:
            j = np.argmax(I)
            L = []
            for i in I:
                if (v.m[i] - v.m[j]) @ invP[i] @ (v.m[i] - v.m[j]) <= self.U:
                    L.append(i)



def generate_model():
    """
    This is the model of the process for the example in "Bayesian Multiple Target Filtering Using Random Finite Sets" by
    Vo, Vo, Clark. The model code is analog to Matlab code provided by
    Vo in http://ba-tuong.vo-au.com/codes.html

    :returns
    - model: dictionary containing the necessary parameters, read through code to understand it better
    """

    model = {}

    # Sampling time, time step duration
    T_s = 1.
    model['T_s'] = T_s

    # number of scans, number of iterations in our simulation
    model['num_scans'] = 100

    # Surveillance region
    x_min = -1000
    x_max = 1000
    y_min = -1000
    y_max = 1000
    model['surveillance_region'] = np.array([[x_min, x_max], [y_min, y_max]])

    # TRANSITION MODEL
    # Probability of survival
    model['p_s'] = 0.99

    # Transition matrix
    I_2 = np.eye(2)
    # F = [[I_2, T_s*I_2], [02, I_2]
    F = np.zeros((4, 4))
    F[0:2, 0:2] = I_2
    F[0:2, 2:] = I_2 * T_s
    F[2:, 2:] = I_2
    model['F'] = F

    # Process noise covariance matrix
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = (T_s ** 4) / 4 * I_2
    Q[0:2, 2:] = (T_s ** 3) / 2 * I_2
    Q[2:, 0:2] = (T_s ** 3) / 2 * I_2
    Q[2:, 2:] = (T_s ** 2) * I_2
    # standard deviation of the process noise
    sigma_w = 5.
    Q = Q * (sigma_w ** 2)
    model['Q'] = Q

    # Parameters for the spawning model: beta(x|ksi) = sum(w[i]*Normal(x,F_spawn[i]*ksi+d_spawn[i],Q_spawn[i]))
    model['w_spawn'] = [0.05]
    model['F_spawn'] = [np.eye(4)]
    model['d_spawn'] = [0]
    Q_spawn = np.eye(4) * 100
    Q_spawn[[2, 3], [2, 3]] = 400
    model['Q_spawn'] = Q_spawn

    # Parameters of the new born targets Gaussian mixture
    w = [0.1, 0.1]
    m = [np.array([250., 250., 0., 0.]), np.array([-250., -250., 0., 0.])]
    P = [np.diag([100., 100, 25, 25]), np.diag([100., 100, 25, 25])]
    model['birth_GM'] = GaussianMixture(w, m, P)

    # MEASUREMENT MODEL
    # probability of detection
    model['p_d'] = 0.98

    # measurement matrix z = Hx + v = N(z; Hx, R)
    model['H'] = np.zeros((2, 4))
    model['H'][:, 0:2] = np.eye(2)
    # measurement noise covariance matrix
    sigma_v = 10  # m
    model['R'] = I_2 * (sigma_v ** 2)

    # the reference to clutter intensity function
    model['lc'] = 50
    model['clutt_int_fun'] = lambda z: clutter_intensity_function(z, model['lc'], model['surveillance_region'])

    # pruning and merging parameters:
    model['T'] = 1e-5
    model['U'] = 4.
    model['Jmax'] = 100

    return model


if __name__ == "__main__":
    print("GMPHD filter")
