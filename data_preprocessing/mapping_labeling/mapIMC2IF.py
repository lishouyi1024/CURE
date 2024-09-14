# Partially adapted from https://github.com/siavashk/pycpd
# Citation: https://joss.theoj.org/papers/10.21105/joss.04681

# Libraries
from __future__ import division
from warnings import warn
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import argparse
import time
import os
import sys
import numbers
mpl.use('tkagg')


# Class
class EMRegistration(object):
    """
    Expectation maximization point cloud registration.

    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    Y: numpy array
        MxD array of source points.

    TY: numpy array
        MxD array of transformed source points.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points

    iteration: int
        The current iteration throughout registration.

    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between 
        source and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective 
        function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector 
        of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    """

    def __init__(self, X, Y, sigma2=None, max_iterations=None, tolerance=None, 
                 w=None, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or 
                                   sigma2 <= 0):
            raise ValueError(
                """Expected a positive value for sigma2 instead 
                got: {}""".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, 
                                                          numbers.Number) or 
                                                          max_iterations < 0):
            raise ValueError("""Expected a positive integer for max_iterations 
                             instead got: {}""".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(
                        max_iterations, int):
            warn("""Received a non-integer value for max_iterations: {}. 
                 Casting to integer.""".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) 
                        or tolerance < 0):
            raise ValueError("""Expected a positive float for tolerance instead 
                            got: {}""".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or 
                        w >= 1):
            raise ValueError("""Expected a value between 0 (inclusive) and 1 
                             (exclusive) for w instead got: {}""".format(w))

        self.X = X
        self.Y = Y
        self.TY = Y
        self.sigma2 = initialize_sigma2(X, Y) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N, ))
        self.P1 = np.zeros((self.M, ))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0

    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        while (self.iteration < self.max_iterations and 
              self.diff > self.tolerance):

            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters(), self.P

    def get_registration_parameters(self):
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        raise NotImplementedError(
            """Updating the source point cloud should be defined in child 
            classes.""")

    def update_variance(self):
        raise NotImplementedError(
            """Updating the Gaussian variance for the mixture model should be 
            defined in child classes.""")

    def iterate(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c
        
        self.P = np.divide(P, den)
        
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)

    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()



# Rigid registration class
class RigidRegistration(EMRegistration):
    """
    Rigid registration.

    Attributes
    ----------
    R: numpy array (semi-positive definite)
        DxD rotation matrix. Any well behaved matrix will do,
        since the next estimate is a rotation matrix.

    t: numpy array
        1xD initial translation vector.

    s: float (positive)
        scaling parameter.

    A: numpy array
        Utility array used to calculate the rotation matrix.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    YPY: float
        Denominator value used to update the scale factor.
        Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    X_hat: numpy array
        Centered target point cloud.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    """

    def __init__(self, R=None, t=None, s=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.D != 2 and self.D != 3:
            raise ValueError(
                """Rigid registration only supports 2D or 3D point clouds. 
                Instead got {}.""".format(self.D))

        if R != None and (R.ndim != 2 or R.shape[0] != self.D or 
                          R.shape[1] is not self.D or 
                          not is_positive_semi_definite(R)):
            raise ValueError(
                """The rotation matrix can only be initialized to {}x{} 
                positive semi definite matrices. Instead got: 
                {}.""".format(self.D, self.D, R))

        if t != None and (t.ndim != 2 or t.shape[0] != 1 or 
                          t.shape[1] != self.D):
            raise ValueError(
                """The translation vector can only be initialized to 1x{} 
                positive semi definite matrices. Instead got: 
                {}.""".format(self.D, t))

        if s != None and (not isinstance(s, numbers.Number) or s <= 0):
            raise ValueError(
                """The scale factor must be a positive number. Instead got: 
                {}.""".format(s))

        self.R = np.eye(self.D) if R is None else R
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
        self.s = 1 if s is None else s

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # target point cloud mean
        muX = np.divide(np.sum(self.PX, axis=0),
                        self.Np)
        # source point cloud mean
        muY = np.divide(
            np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.X_hat = self.X - np.tile(muX, (self.N, 1))
        # centered source point cloud
        Y_hat = self.Y - np.tile(muY, (self.M, 1))
        self.YPY = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(Y_hat, Y_hat), axis=1))

        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        # Singular value decomposition as per lemma 1 of 
        # https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))

        # Calculate the rotation matrix using Eq. 9 of 
        # https://arxiv.org/pdf/0905.2635.pdf.
        self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
        # Update scale and translation using Fig. 2 of 
        # https://arxiv.org/pdf/0905.2635.pdf.
        self.s = np.trace(np.dot(np.transpose(self.A), ##### Changed here
                                 np.transpose(self.R))) / self.YPY
        self.t = np.transpose(muX) - self.s * \
            np.dot(np.transpose(self.R), np.transpose(muY))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the rigid transformation.

        """
        if Y is None:
            self.TY = self.s * np.dot(self.Y, self.R) + self.t
            return
        else:
            return self.s * np.dot(Y, self.R) + self.t

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the 
        rigid transformation. See the update rule for sigma2 in Fig. 2 of of 
        https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.q

        trAR = np.trace(np.dot(self.A, self.R))
        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X_hat, self.X_hat), axis=1))
        self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / \
            (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.diff = np.abs(self.q - qprev)
        self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Return the current estimate of the rigid transformation parameters.

        """
        return self.s, self.R, self.t


# Function
def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError(
            """Encountered an error while checking if the matrix is positive 
            semi definite. Expected a numpy array, instead got : {}""".format(R))
    return np.all(np.linalg.eigvals(R) > 0)


def initialize_sigma2(X, Y):
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)


def lowrankQS(G, beta, num_eig, eig_fgt=False):
    # if we do not use FGT we construct affinity matrix G and find the
    # first eigenvectors/values directly

    if eig_fgt is False:
        S, Q = np.linalg.eigh(G)
        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:, eig_indices]  # eigenvectors
        S = S[eig_indices]  # eigenvalues.

        return Q, S

    elif eig_fgt is True:
        raise Exception('Fast Gauss Transform Not Implemented!')


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='X')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Y')
    plt.text(0.87, 0.92, 'Iteration: {:d}'.format(iteration),
             horizontalalignment='center', verticalalignment='center',
             transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main(args):
    
    df_x = pd.read_table(args.x, sep='\t')
    df_y = pd.read_table(args.y, sep='\t')
    X = df_x.loc[:, ['x', 'y']].to_numpy()
    Y = df_y.loc[:, ['x', 'y']].to_numpy()

    #X = np.loadtxt(args.x) 
    #Y = np.loadtxt(args.y)
    
    reg = RigidRegistration(
         **{'X': X, 'Y': Y, 'max_iterations': args.max_iter,
            'tolerance': args.tol, 'w': args.w})

    # output = reg.register()

    # Visualizing output (X: red, Y: blue)
    # ---------------------------------------------
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])
    output = reg.register(callback)
    plt.show()
    
    print(f"{reg.iteration}\t{reg.diff}\t{reg.P.shape}")
    print(f"Transformation parameters:\n"
          f"- Scale:\n{output[1][0]}\n"
          f"- Rotation:\n{output[1][1]}\n"
          f"- Translation:\n{output[1][2]}")
    
    Y_trans = reg.transform_point_cloud(Y=Y)
    df_y.loc[:, ['x', 'y']] = Y_trans
    df_y.to_csv(args.output, index=False, sep='\t')
    #np.savetxt(args.output, Y_trans)
    
    print(X[:10, :10])
    print(Y_trans[:10, :10])
    print(Y[:10, :10])



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='finds the transformation between 2 point sets',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-x', type=str, required=True,
        help="""path to tab-delimited (x,y) coordinate file of target (fixed) 
                points""")

    parser.add_argument(
        '-y', type=str, required=True,
        help="""path to tab-delimited (x,y) coordinate file of source (moving) 
                points""")

    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help="path to tab-delimited transformed (x,y) coordinates of source")

    parser.add_argument(
        '--max_iter', type=int, default=700,
        help="""Iterative registration will terminate once the algorithm has 
                taken this many iterations.""")

    parser.add_argument(
        '--tol', type=float, default=1e-4,
        help="""Registration will terminate once the difference between 
                consecutive objective function values falls within this 
                tolerance.""")

    parser.add_argument(
        '-w', type=float, default=1e-3,
        help="""Contribution of the uniform distribution to account for 
                outliers. Valid values span 0 (inclusive) and 1 (exclusive).""")

    args = parser.parse_args()
    main(args)
