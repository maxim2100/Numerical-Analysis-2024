@staticmethod
def machine_epsilon():
    """Computes machine epsilon using a simple iterative method."""
    eps = 1.0
    while (1.0 + eps) > 1.0:
        eps /= 2.0
    return eps * 2.0