import numpy as np

class Simulation:
    def __init__(self, n=500, p=300, rho_star=(0.9, 0.8), seed=None):
        self.n = n
        self.p = p
        self.q = p
        self.rho_star = rho_star
        self.rng = np.random.default_rng(seed)

    def _find_orthonormal_alpha(self, alpha1, alpha2_init, Sigma):
        """
        Need to confirm with prof:
        https://ubcmath.github.io/MATH307/orthogonality/projection.html
        https://dept.math.lsa.umich.edu/~speyer/417/OrthoProj.pdf
        Refer to David C. Lay's Linear Algebra and its Applications p.349, extended to Mahalanobis distance 
        """
        projection = (alpha1 @ Sigma @ alpha2_init) / (alpha1 @ Sigma @ alpha1) * alpha1
        alpha2_orthogonal = alpha2_init - projection
        alpha2_orthonormal = self._normalize(alpha2_orthogonal, Sigma)
        return alpha2_orthonormal
    
    def _normalize(self, v, Sigma):
        return v / np.sqrt(v.T @ Sigma @ v)

    def _build_cross_covariance(self, alphas, betas, rhos, Sigma_YY, Sigma_XX):
        return Sigma_YY @ sum(r * np.outer(a, b) for r, a, b in zip(rhos, alphas, betas)) @ Sigma_XX

    def _build_joint_covariance(self, Sigma_YY, Sigma_XX, alphas, betas, rhos):
        Sigma_YX = self._build_cross_covariance(alphas, betas, rhos, Sigma_YY, Sigma_XX)
        Sigma_XY = Sigma_YX.T
        return np.block([[Sigma_YY, Sigma_YX], [Sigma_XY, Sigma_XX]])

    def _generate_data(self, Sigma):
        """
        Y is of shape n x p
        X is of shape n x q 
        Our notations assumes Z = [Y], of shape n x (p+q)
                                  [X]
        """
        Z = self.rng.multivariate_normal(np.zeros(self.p + self.q), Sigma, size=self.n)
        return Z[:, :self.p], Z[:, self.p:] # :self.p --> p, self.p: --> q

    def _generate_groundtruth(self, eta1, eta2, Sigma_YY, Sigma_XX):
        self.Sigma_YY = Sigma_YY
        self.Sigma_XX = Sigma_XX

        alpha1 = self._normalize(eta1, Sigma_YY)
        beta1 = self._normalize(eta1, Sigma_XX)

        if eta2 is not None: # two canonical weight vectors, K = 2 
            alpha2 = self._normalize(eta2, Sigma_YY)
            alpha2 = beta2 = self._find_orthonormal_alpha(alpha1, alpha2, Sigma_YY)

            Sigma = self._build_joint_covariance(Sigma_YY, Sigma_XX,
                                                [alpha1, alpha2], [beta1, beta2],
                                                self.rho_star)
            A_star = np.column_stack([alpha1, alpha2])
            B_star = np.column_stack([beta1, beta2])
            rho_star = self.rho_star
        else: # one canonical weight vectors, K = 1
            Sigma = self._build_joint_covariance(Sigma_YY, Sigma_XX,
                                                [alpha1], [beta1],
                                                [self.rho_star[0]])
            A_star = alpha1[:, None]
            B_star = beta1[:, None]
            rho_star = (self.rho_star[0],)

        Y, X = self._generate_data(Sigma)
        return Y, X, A_star, B_star, rho_star

    def model1(self): 
        """
        Identity covariances
        """
        idx1 = [0, 1, 2, 3]
        idx2 = [5, 6, 7, 8]
        eta1 = np.zeros(self.p); eta1[idx1] = [1, 1, 1, 1]
        eta2 = np.zeros(self.p); eta2[idx2] = [1, 1, 1, 1]
        Sigma_YY = Sigma_XX = np.eye(self.p)
        return self._generate_groundtruth(eta1, eta2, Sigma_YY, Sigma_XX)
    
    def model2(self): 
        """
        Moderate correlation, AR(0.3)
        """
        rho = 0.3
        idx1 = [0, 1, 2, 3]
        idx2 = [5, 6, 7, 8]
        eta1 = np.zeros(self.p); eta1[idx1] = [1, 1, 1, 1]
        eta2 = np.zeros(self.p); eta2[idx2] = [1, 1, 1, 1]
        Sigma_YY = Sigma_XX = rho ** np.abs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return self._generate_groundtruth(eta1, eta2, Sigma_YY, Sigma_XX)
    
    def model3(self):
        """
        High correlation, AR(0.8)
        """ 
        rho = 0.8
        idx1 = [0, 1, 2, 3]
        idx2 = [5, 6, 7, 8]
        eta1 = np.zeros(self.p); eta1[idx1] = [1, 1, 1, 1]
        eta2 = np.zeros(self.p); eta2[idx2] = [1, 1, 1, 1]
        Sigma_YY = Sigma_XX = rho ** np.abs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return self._generate_groundtruth(eta1, eta2, Sigma_YY, Sigma_XX)
    
    def model4(self):
        """
        Sparse precision matrices
        """
        idx1 = [0, 1, 2, 3]
        idx2 = [5, 6, 7, 8]
        eta1 = np.zeros(self.p); eta1[idx1] = [1, 1, 1, 1]
        eta2 = np.zeros(self.p); eta2[idx2] = [1, 1, 1, 1]
        Omega = np.eye(self.p)
        for i in range(self.p):
            if i + 1 < self.p: Omega[i, i+1] = Omega[i+1, i] = 0.5
            if i + 2 < self.p: Omega[i, i+2] = Omega[i+2, i] = 0.4
        Sigma0 = np.linalg.inv(Omega)
        Lambda = np.diag(1.0 / np.sqrt(np.diag(Sigma0)))
        Sigma_YY = Sigma_XX = Lambda @ Sigma0 @ Lambda
        return self._generate_groundtruth(eta1, eta2, Sigma_YY, Sigma_XX)
    
    def model5(self):
        """
        Identity covariances with 1 sparse canonical pair (first 4 nonzero)
        """
        idx = [0, 1, 2, 3]
        eta = np.zeros(self.p); eta[idx] = [1] * 4
        Sigma_YY = Sigma_XX = np.eye(self.p)
        return self._generate_groundtruth(eta, None, Sigma_YY, Sigma_XX)

    def model6(self):
        """
        Moderate correlation with 1 sparse canonical pair (first 8 nonzero), AR(0.5)
        """
        rho = 0.5
        idx = [0, 1, 2, 3, 4, 5, 6, 7]
        eta = np.zeros(self.p); eta[idx] = [1] * 8
        Sigma_YY = Sigma_XX = rho ** np.abs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return self._generate_groundtruth(eta, None, Sigma_YY, Sigma_XX)

    def model7(self): 
        """
        Moderate correlation with 2 sparse canonical pairs, AR(0.5)
        """
        rho = 0.5
        idx1 = [0, 1, 2, 3]
        idx2 = [5, 6, 7, 8]
        eta1 = np.zeros(self.p); eta1[idx1] = [1, 1, 1, 1]
        eta2 = np.zeros(self.p); eta2[idx2] = [1, 1, 1, 1]
        Sigma_YY = Sigma_XX = rho ** np.abs(np.subtract.outer(np.arange(self.p), np.arange(self.p)))
        return self._generate_groundtruth(eta1, eta2, Sigma_YY, Sigma_XX)
    
    def model8(self):
        """
        Moderate correlation with 2 sparse canonical pairs, CS(0.5)
        """
        rho = 0.5
        idx1 = [0, 1, 2, 3]
        idx2 = [5, 6, 7, 8]
        eta1 = np.zeros(self.p); eta1[idx1] = [1, 1, 1, 1]
        eta2 = np.zeros(self.p); eta2[idx2] = [1, 1, 1, 1]
        Sigma_YY = Sigma_XX = rho * np.ones((self.p, self.p)) + (1 - rho) * np.eye(self.p)
        return self._generate_groundtruth(eta1, eta2, Sigma_YY, Sigma_XX)