import numpy as np
import pandas as pd
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.fixture
def df_input():
    return pd.DataFrame(
        {
            "A":np.random.normal(0,1,1000),
            "B":np.random.normal(10,100,1000),
            "C":np.random.randint(0,10000,1000)
        }
    )


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5), (0.25,0.75), (0.3, 0.7)]
)

def test_winsorizer(df_input, lower_quantile, upper_quantile):
    
    test_winsorizer = Winsorizer(lower_quantile, upper_quantile)
    df_output = test_winsorizer.fit_transform(df_input)

    X = df_input.to_numpy()
    df_lower_bound = np.quantile(X, lower_quantile, axis=0)
    df_upper_bound = np.quantile(X, upper_quantile, axis=0)

    assert np.all(df_output >= df_lower_bound)
    assert np.all(df_output <= df_upper_bound)
    assert df_output.shape == X.shape
