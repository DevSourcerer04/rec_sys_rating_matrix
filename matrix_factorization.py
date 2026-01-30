import numpy as np


def generate_random_factors(num_users, num_items, k, seed=None, low=0.0, high=1.0):
    """
    Returns P (num_users x k) and Q (num_items x k) with random values.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(low, high, size=(num_users, k))
    q = rng.uniform(low, high, size=(num_items, k))
    return p, q


def predict_ratings(p, q):
    """
    Returns R_hat = P @ Q.T (num_users x num_items).
    """
    return p @ q.T


def total_squared_error(r, p, q, mask=None):
    """
    Computes E = sum_{(i,j) in T} (r_ij - p_i^T q_j)^2.
    If mask is None, uses all entries.
    """
    r_hat = predict_ratings(p, q)
    if mask is None:
        diff = r - r_hat
        return np.sum(diff * diff)
    diff = (r - r_hat)[mask]
    return np.sum(diff * diff)


def sgd_step(r, p, q, alpha, mask=None):
    """
    One SGD pass over observed entries using:
      e_ij = r_ij - p_i^T q_j
      p_ik' = p_ik + 2 * alpha * e_ij * q_kj
      q_kj' = q_kj + 2 * alpha * e_ij * p_ik
    """
    num_users, num_items = r.shape
    for i in range(num_users):
        for j in range(num_items):
            if mask is not None and not mask[i, j]:
                continue
            e_ij = r[i, j] - np.dot(p[i], q[j])
            p_i_old = p[i].copy()
            p[i] = p[i] + 2.0 * alpha * e_ij * q[j]
            q[j] = q[j] + 2.0 * alpha * e_ij * p_i_old


def demo():
    # Example usage
    num_users, num_items, k = 4, 5, 3
    p, q = generate_random_factors(num_users, num_items, k, seed=42)
    r_hat = predict_ratings(p, q)

    # Create a noisy "observed" R from the generated R_hat
    rng = np.random.default_rng(123)
    r = r_hat + rng.normal(0.0, 0.1, size=r_hat.shape)

    print("Initial error:", total_squared_error(r, p, q))
    sgd_step(r, p, q, alpha=0.01)
    print("Error after 1 SGD step:", total_squared_error(r, p, q))


if __name__ == "__main__":
    demo()
