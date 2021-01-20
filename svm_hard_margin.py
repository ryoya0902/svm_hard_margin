# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class SVCHard:
    def __init__(self, iterarions):
        self.iterations = iterarions
        self._alphas = None
        self.coef_ = None

    def fit(self, X_train, y_train):
        count = 0
        result = []
        self._alphas = np.zeros(X_train.shape[0])
        while count < self.iterations:
            f_i, idx_i = self._calc_f_idx(X_train, y_train, optimize="min")
            f_j, idx_j = self._calc_f_idx(X_train, y_train, optimize="max")
            if count != 0 and (y_train[idx_i] * f_i) > (y_train[idx_i] * f_i):
                break
            alpha_i, alpha_j = self._calc_alpha(X_train, y_train, idx_i, idx_j)
            pre_alphas = self._alphas.copy()
            self._alphas[idx_i] = alpha_i
            self._alphas[idx_j] = alpha_j
            if np.array_equal(pre_alphas, self._alphas):
                break
            w = []
            for d in range(X_train.shape[1]):
                w_d = 0
                for n in range(X_train.shape[0]):
                    w_d += self._alphas[n] * y_train[n] * X_train[n][d]
                w.append(w_d)
            w_0 = 0
            for n in range(X_train.shape[0]):
                w_0 += y_train[n]
                for m in range(X_train.shape[0]):
                    w_0 -= self._alphas[m] * y_train[m] * (X_train[n] @ X_train[m])
            w_0 /= X_train.shape[0]
            count += 1
        w = np.array(w)
        for i in range(X_train.shape[0]):
            y = w_0 + w @ X_train[i]
            if y >= 0:
                result.append(1)
            else:
                result.append(-1)
        result = np.array(result)
        self.coef_ = np.hstack([w_0, w])
        return result

    def predict(self, X_test):
        result = []
        w_0 = self.coef_[0]
        w = self.coef_[1:]
        for i in range(X_test.shape[0]):
            y = w_0 + w @ X_test[i]
            if y >= 0:
                result.append(1)
            else:
                result.append(-1)
        result = np.array(result)
        return result

    def _calc_alpha(self, X_train, y_train, idx_i, idx_j):
        sum_ = 0
        a_t = 0
        for i in range(X_train.shape[0]):
            if i == idx_i:
                continue
            else:
                f = X_train[idx_j] * self._alphas[i] * y_train[i]
                s = self._alphas[i] * y_train[i] * X_train[i]
                sum_ += f - s
                a_t += self._alphas[i] * y_train[i]
        x_ = (
            1
            - y_train[idx_i] * y_train[idx_j]
            + y_train[idx_i] * ((X_train[idx_i] - X_train[idx_j]) @ sum_)
        )

        y_ = (X_train[idx_i] - X_train[idx_j]) @ (X_train[idx_i] - X_train[idx_j])
        alpha_i = x_ / y_
        if alpha_i < 0:
            alpha_i = 0
        alpha_j = y_train[idx_j] * (-1 * alpha_i * y_train[idx_i] - a_t)
        if alpha_j < 0:
            alpha_j = 0
        return alpha_i, alpha_j

    def _calc_f_idx(self, X_train, y_train, optimize):
        if optimize == "min":
            t_i = np.where(y_train == -1)[0]
            best_f = 10 * 8
            best_idx = None
        else:
            t_i = np.where(y_train == 1)[0]
            best_f = -10 * 8
            best_idx = None
        for k in t_i:
            f = 1
            for i in range(X_train.shape[0]):
                f -= (
                    self._alphas[i]
                    * y_train[k]
                    * y_train[i]
                    * (X_train[k] @ X_train[i])
                )
            if best_f > f and optimize == "min":
                best_f = f
                best_idx = k
            if best_f < f and optimize == "max":
                best_f = f
                best_idx = k
        return best_f, best_idx


if __name__ == "__main__":
    ITERATIONS = 10
    SEED = 42

    digits = load_digits(n_class=2)
    X = digits.data
    y = digits.target
    y = np.where(y == 0, -1, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=SEED
    )

    svc_hard = SVCHard(ITERATIONS)

    train_result = svc_hard.fit(X_train, y_train)
    train_accuracy = np.sum(train_result == y_train) / len(y_train)
    print(f"train_accuray: {train_accuracy}")

    test_result = svc_hard.predict(X_test)
    test_accuracy = np.sum(test_result == y_test) / len(y_test)
    print(f"test_accuray: {test_accuracy}")