import numpy as np
import invsqrt

class PegasosSVM:
    def __init__(self, lam=1e-4, T=10000):
        self.lam = lam
        self.T = T
        self.w = None

    def fit(self, X, y, norm = False):
        n, d = X.shape
        self.w = np.zeros(d)

        for t in range(1, self.T + 1):
            i = np.random.randint(0, n)
            x_i = X[i]
            y_i = y[i]

            eta_t = 1.0 / (self.lam * t)

            if y_i * (np.dot(self.w, x_i)) < 1:
                self.w = (1 - eta_t * self.lam) * self.w + eta_t * y_i * x_i
            else:
                self.w = (1 - eta_t * self.lam) * self.w

            if norm:
                norm_w = np.linalg.norm(self.w)
                factor = min(1.0, 1.0 / (np.sqrt(self.lam) * norm_w))
                self.w = factor * self.w
            print(self.w)

    def predict(self, X, _, _2):
        return np.sign(X @ self.w)


def linear_kernel(x, y):
    return np.dot(x, y)


class PegasosSVMKernel:
    def __init__(self, lam=1e-4, T=10000, kernel=None):
        self.lam = lam
        self.T = T
        self.a = None
        self.kernel = kernel if kernel is not None else linear_kernel

    def fit(self, X, y):
        n = X.shape[0]
        self.a = np.zeros(n)

        for t in range(1, self.T + 1):
            i = np.random.randint(0, n)
            x_i = X[i]
            y_i = y[i]

            K_vals = np.array([self.kernel(x_i, X[j]) for j in range(n)])
            sum_ = np.sum(self.a * y * K_vals)

            if (1 / (self.lam * t)) * y_i * sum_ < 1:
                self.a[i] = self.a[i] + 1

    def predict(self, X_test, X_train, y_train):
        preds = []
        for x in X_test:
            s = np.sum(self.a * y_train * np.array([self.kernel(x, X_train[j]) for j in range(len(X_train))]))
            preds.append(np.sign(s))
        return np.array(preds)

class PegasosSVMBatch:
    def __init__(self, lam=1e-4, T=10000, k=5):
        self.lam = lam
        self.T = T
        self.w = None
        self.k = k

    def fit(self, X, y, norm = False):
        n, d = X.shape
        self.w = np.zeros(d)
        m = range(0, n)
        for t in range(1, self.T + 1):
            A_t = np.random.choice(m, size=self.k, replace=False)
            A_t_plus = [i for i in A_t if y[i] * np.dot(self.w, X[i]) < 1]

            eta_t = 1.0 / (self.lam * t)

            self.w = (1 - eta_t * self.lam) * self.w + (eta_t / self.k) * sum(y[i] * X[i] for i in A_t_plus)

            if norm:
                norm_w = np.linalg.norm(self.w)
                factor = min(1.0, 1.0 / (np.sqrt(self.lam) * norm_w))
                self.w = factor * self.w

    def predict(self, X, _, _2):
        return np.sign(X @ self.w)


if __name__ == "__main__":
    np.random.seed(0)
    n = 500
    d = 4
    X = np.random.randn(n, d)


    # Create a random linear separator
    true_w = np.random.randn(d)
    y = np.sign(X @ true_w)


    svm = PegasosSVMBatch(lam=0.05, T=20, k=5)
    svm.fit(X, y)

    preds = svm.predict(X, X, y)
    accuracy = np.mean(preds == y)
    print("Training accuracy:", accuracy)
