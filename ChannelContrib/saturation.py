from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np

class ExponentialSaturation(BaseEstimator, TransformerMixin):

    """
    Apply exponential saturation.
    The formula is 1 - exp(-exponent * x).
    Parameters
    ----------
    exponent : float, default=1.0
        The exponent.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1000], [2, 1000], [3, 1000]])
    >>> ExponentialSaturation().fit_transform(X)
    array([[0.63212056, 1.        ],
            [0.86466472, 1.        ],
            [0.95021293, 1.        ]])
    """

    def __init__(self, a=1.):
        self.a = a
        
    def fit(self, X, y=None):
        X = check_array(X)

        self._check_n_features(X, reset=True) # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)

        X = check_array(X)
        self._check_n_features(X, reset=False) # from BaseEstimator
        return 1 - np.exp(-self.a*X)


class BoxCoxSaturation(BaseEstimator, TransformerMixin):
    """
    Apply the Box-Cox saturation.
    The formula is ((x + shift) ** exponent - 1) / exponent if exponent != 0,
    else ln(x + shift).

    Parameters
    ----------
    exponent: float, default=1.0
        The exponent.
    shift: float, optional
        The shift. If not provided, it will be dynamically set to ensure no negative values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1000], [2, 1000], [3, 1000]])
    >>> BoxCoxSaturation(exponent=0.5).fit_transform(X)
    array([[ 0.82842712, 61.27716808],
           [ 1.46410162, 61.27716808],
           [ 2.        , 61.27716808]])
    """

    def __init__(self, exponent: float = 1.0, shift: float = None):
        self.exponent = exponent
        self.shift = shift

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)  # from BaseEstimator

        min_value = np.min(X)
        if min_value < 1:
            self.shift_ = 1 - min_value
        else:
            self.shift_ = 0
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        # Compute the values to be transformed
        X_shifted = X + self.shift_

        # Define constants
        epsilon = 1e-8
        min_positive = 1e-8  # Minimum positive value to ensure strict positivity

        # Apply the transformation
        if self.exponent != 0:
            # For values where X + shift > 0, use the Box-Cox formula
            positive_mask = X_shifted > 0
            transformed = np.where(
                positive_mask,
                ((X_shifted) ** self.exponent - 1) / self.exponent,
                ((np.clip(X_shifted + epsilon, a_min=min_positive, a_max=None)) ** self.exponent - 1) / self.exponent
            )
        else:
            # For values where X + shift > 0, use the Box-Cox formula
            positive_mask = X_shifted > 0
            transformed = np.where(
                positive_mask,
                np.log(X_shifted),
                np.log(np.clip(X_shifted + epsilon, a_min=min_positive, a_max=None))
            )

        return transformed


class HillSaturation(BaseEstimator, TransformerMixin):

    """
    Apply the Hill saturation.
    The formula is 1 / (1 + (half_saturation / x) ** exponent).
    Parameters
    ----------
    exponent : float, default=1.0
    The exponent.
    half_saturation : float, default=1.0
    The point of half saturation, i.e. Hill(half_saturation) = 0.5.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1000], [2, 1000], [3, 1000]])
    >>> HillSaturation().fit_transform(X)
    array([[0.5       , 0.999001  ],
        [0.66666667, 0.999001  ],
        [0.75      , 0.999001  ]])
    """

    def __init__(self,   exponent: float = 1.0, half_saturation: float = 1.0):
        self.exponent = exponent
        self.half_saturation = half_saturation
        
    def fit(self, X, y=None):

        X = check_array(X)
        self._check_n_features(X, reset=True) # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        eps = np.finfo(np.float64).eps
        return 1 / (1 + (self.half_saturation / (X + eps)) ** self.exponent)


class AdbudgSaturation(BaseEstimator, TransformerMixin):

    """
    Apply the Adbudg saturation.
    The formula is x ** exponent / (denominator_shift + x ** exponent).
    Parameters
    ----------
    exponent : float, default=1.0
    The exponent.
    denominator_shift : float, default=1.0
    The shift in the denominator.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1000], [2, 1000], [3, 1000]])
    >>> AdbudgSaturation().fit_transform(X)
    array([[0.5       , 0.999001  ],
    [0.66666667, 0.999001  ],
    [0.75      , 0.999001  ]])
    """

    def __init__(self,   exponent: float = 1.0, denominator_shift: float = 1.0):
        self.exponent = exponent
        self.denominator_shift = denominator_shift
        
    def fit(self, X, y=None):

        X = check_array(X)
        self._check_n_features(X, reset=True) # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return X**self.exponent / (self.denominator_shift + X**self.exponent)