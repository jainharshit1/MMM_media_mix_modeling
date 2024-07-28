
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from scipy.signal import convolve2d
import numpy as np

class ExponentialCarryover(BaseEstimator, TransformerMixin):

    """
    Smoothes time series data with an exponential window.
    Smooth the columns of an array by applying a convolution with an exponentially
    decaying curve. This class can be used for modelling carry over effects in
    marketing mix models.
    Parameters
    ----------
    window : int, default=1
    Size of the sliding window. The effect of a holiday will reach from
    approximately date - `window/2 * frequency` to date + `window/2 * frequency`,
    i.e. it is centered around the dates in `dates`.
    strength : float, default=0.0
    Fraction of the spending effect that is carried over.
    peak : float, default=0.0
    Where the carryover effect peaks.
    exponent : float, default=1.0
    To further widen or narrow the carryover curve. A value of 1.0 yields a normal
    exponential decay. With values larger than 1.0, a super exponential decay can
    be achieved.
    mode : str
        Which convolution mode to use. Can be one of
            - "full": The output is the full discrete linear convolution of the inputs.
            - "valid": The output consists only of those elements that do not rely on
            the zero-padding.
            - "same": The output is the same size as the first input, centered with
            respect to the 'full' output.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)
    >>> ExponentialCarryover().fit_transform(X)
    array([[0.],
        [0.],
        [0.],
        [1.],
        [0.],
        [0.],
        [0.]])
    >>> ExponentialCarryover(window=3, strength=0.5).fit_transform(X)
    array([[0.        ],
        [0.        ],
        [0.        ],
        [0.57142857],
        [0.28571429],
        [0.14285714],
        [0.        ]])
    >>> ExponentialCarryover(window=3, strength=0.5, peak=1).fit_transform(X)
    array([[0.  ],
        [0.  ],
        [0.  ],
        [0.25],
        [0.5 ],
        [0.25],
        [0.  ]])
    """

    def __init__(self, strength=0.5, length=1):
        self.strength = strength
        self.length = length

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window_ = (
                self.strength ** np.arange(self.length + 1)
        ).reshape(-1, 1)
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window_)
        if self.length > 0:
            convolution = convolution[: -self.length]

        inflation_total = np.sum(convolution) / np.sum(X)
        sliding_window = self.sliding_window_.flatten()

        result= {
            'x': X,
            'x_decayed': convolution,
            'sliding_window_cum': sliding_window,
            'inflation_total': inflation_total
        }
        # print("Decay rate is as follows: ",result['sliding_window_cum'])
        return result



class GaussianCarryover(BaseEstimator, TransformerMixin):

    """
    Smoothes time series data with a Gaussian window.
    Smooth the columns of an array by applying a convolution with a generalized
    Gaussian curve.
    Parameters
    ----------
    window : int, default=1
    Size of the sliding window. The effect of a holiday will reach from
    approximately date - `window/2 * frequency` to date + `window/2 * frequency`,
    i.e. it is centered around the dates in `dates`.
    p : float, default=1
    Parameter for the shape of the curve. p=1 yields a typical Gaussian curve
    while p=0.5 yields a Laplace curve, for example.
    sig : float, default=1
    Parameter for the standard deviation of the bell-shaped curve.
    tails : str, default="both"
    Which tails to use. Can be one of
    - "left"
    - "right"
    - "both"
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)
    >>> GaussianCarryover().fit_transform(X)
    array([[0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.]])
    >>> GaussianCarryover05448868(window=5, p=1, sig=1).fit_transform(X)
    array([[0.        ],
    [0.05448868],
    [0.24420134],
    [0.40261995],
    [0.24420134],
    [0.],
    [0.        ]])
    >>> GaussianCarryover(window=7, tails="right").fit_transform(X)
    array([[0.        ],
    [0.        ],
    [0.        ],
    [0.57045881],
    [0.34600076],
    [0.0772032 ],
    [0.00633722]])
    """

    def __init__(self,window: int = 2,p: float = 1, sig: float = 1, tails: str = "both",mode: str = "full"):
        self.window=window
        self.p = p
        self.sig = sig
        self.tails = tails
        self.mode = mode

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window = np.exp(
            -0.5
            * np.abs(np.arange(-self.window // 2 + 1, self.window // 2 + 1) / self.sig)
            ** (2 * self.p)
        )
        if self.tails == "left":
            self.sliding_window[self.window // 2 + 1 :] = 0
        elif self.tails == "right":
            self.sliding_window[: self.window // 2] = 0
        elif self.tails != "both":
            raise ValueError(
                "tails keyword has to be one of 'both', 'left' or 'right'."
            )

        self.sliding_window = (
            self.sliding_window.reshape(-1, 1) / self.sliding_window.sum()
        )
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window,mode=self.mode)
        if self.mode == "full" and self.window > 1:
            convolution = convolution[: -self.window+1]
        # import pdb;pdb.set_trace()
        inflation_total = np.sum(convolution) / np.sum(X)
        sliding_window = self.sliding_window.flatten()
        result= {
            'x': X,
            'x_decayed': convolution,
            'sliding_window_cum': sliding_window,
            'inflation_total': inflation_total
        }
        # print("decay rate is as follows: ",result['sliding_window_cum'])
        return result


# X = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)
# model = ExponentialCarryover(length=3, strength=0.5)
# result = model.fit_transform(X)
# print(result)

# X = np.array([100, 0, 0, 0, 100, 0, 0,0,0]).reshape(-1, 1)
# result = GaussianCarryover(window=5, p=1, sig=1).fit_transform(X)
# print(result)