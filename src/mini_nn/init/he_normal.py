import numpy as np

def he_normal(in_features: int, out_features: int):
    '''
    Generates gaussian He initialization.

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
        std = np.sqrt(2/in_features)
        weights = gen.normal(loc=0, scale=std, size=(in_features, out_features))
        ```
    '''

    # --- set random generator ---
    gen = np.random.default_rng()

    # --- calculate he normal weights ---
    std = np.sqrt(2/in_features)
    weights = gen.normal(loc=0, scale=std, size=(in_features, out_features))

    return weights