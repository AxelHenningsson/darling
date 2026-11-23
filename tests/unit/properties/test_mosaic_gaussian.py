import numpy as np
import pytest

from darling import io, properties


@pytest.mark.parametrize(
    "power, expected",
    [
        (6, [0.26422507, 8.004202, 0.7280357, 0.70530385, 0.6971552, 0.27307388]),
        (1, [-0.07137229, 7.945401, 0.7149295, 0.77881926, 1.1805742, 0.5295156]),
    ],
)
def test_matches_original_domains_fit(power, expected):
    # Independent outputs from the fitting branch (b7ba50a), scan 1.1.
    _, data, grid = io.assets.domains()
    center, cov, error = properties.fit_gaussian(data[4:5, 124:125], grid, power)
    upper = cov[0, 0][np.triu_indices(2)]
    np.testing.assert_allclose(
        [*center[0, 0], *upper, error.item()], expected, rtol=2e-5, atol=2e-6
    )


def test_parameters_and_moment_seeds():
    x, y = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-2, 2, 29), indexing="ij")
    grid = np.stack([x, y])
    sx, sy, rho = 0.32, 0.4, -0.3
    z = (x / sx) ** 2 + (y / sy) ** 2 - 2 * rho * x * y / (sx * sy)
    data = (1000 * np.exp(-z / (2 * (1 - rho**2))))[None, None]
    # Seeding at the analytical solution isolates the model convention,
    # covariance packing and unweighted full-scan error calculation.
    mu = np.zeros((1, 1, 2), dtype=np.float32)
    seed = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])[None, None]
    kwargs = dict(weight_power=6, stat_mean=mu, stat_cov=seed)
    params = properties.curvefit.fit_2D_gaussian(data, grid, **kwargs)[0, 0]
    center, cov, error = properties.fit_gaussian(data, grid, **kwargs)
    assert params[6] == 1
    np.testing.assert_allclose(
        params[:6], [1000, 0, 0, sx, sy, rho], atol=0.012, rtol=0.04
    )
    np.testing.assert_array_equal(center[0, 0], params[1:3].astype(np.float32))
    _, mx, my, a, b, r = params[:6]
    expected_cov = np.array([[a * a, r * a * b], [r * a * b, b * b]], dtype=np.float32)
    np.testing.assert_array_equal(cov[0, 0], expected_cov)
    dx, dy = (x - mx) / a, (y - my) / b
    pred = params[0] * np.exp(-0.5 * (dx**2 + dy**2 - 2 * r * dx * dy) / (1 - r**2))
    np.testing.assert_allclose(
        error.item(), np.sqrt(np.mean((pred - data) ** 2)) / data.max(), atol=1e-7
    )


def test_mask_and_unusable_pixels():
    grid = np.indices((3, 4), dtype=np.float32)
    data = np.zeros((1, 6, 3, 4), dtype=np.float32)
    data[0, 1] = 2
    data[0, 2, 1, 2] = np.nan
    data[0, 3, 1, 2] = -1
    data[0, 4:, 1:, 1:] = 10
    mask = np.array([[True, True, True, True, False, True]])
    fit = properties.curvefit.fit_2D_gaussian(data, grid, mask=mask)
    np.testing.assert_array_equal(fit[0, :, 6], [0, 0, 0, 0, 0, 1])
    _, _, error = properties.fit_gaussian(data, grid, mask=mask)
    assert error[0, 0] == 0
    assert np.isnan(error[0, 1:5]).all()
    assert np.isfinite(error[0, 5])
    with pytest.raises(ValueError, match="two motor grids"):
        properties.curvefit.fit_2D_gaussian(data, grid[:1])
