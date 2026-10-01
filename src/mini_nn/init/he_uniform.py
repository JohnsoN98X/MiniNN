import numpy as np

def he_uniform(in_features: int, out_features: int):
    '''
    Generates uniform He initialization.

    Parameters
    ----------
    in_features: int
        number of input features. should be passed from the layer by "in_features". 
    out_features: int
        number of output features. should be passed from the layer by "out_features".

    Returns
    ----------
    weights: np.ndarray
        weights matrix, initialized by Gaussian He method. calcualted as:
        ```python
        std = np.sqrt(6/in_features)
        weights = gen.uniform(-std, std, size=(in_features, out_features))
        ```
    '''

    # --- set random generator ---
    gen = np.random.default_rng()

    # --- calculate he normal weights ---
    std = np.sqrt(6/in_features)
    weights = gen.uniform(-std, std, size=(in_features, out_features))

    return weights