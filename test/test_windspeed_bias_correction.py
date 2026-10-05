# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
# SPDX-License-Identifier: MIT
"""
Regression contracts for bias correction.

Failures: supplied means lose the CRS during feature preparation; unsupported
modules prevent selection of ERA5; dimension order scrambles raster values;
annual means weight leap and ordinary years equally; skipped modules stop
preparation; successive modules lose prepared-feature metadata.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio as rio
import xarray as xr
from atlite import Cutout
from atlite.datasets import era5
from atlite.wind import calculate_windspeed_bias_correction
from rasterio.transform import from_origin


@pytest.fixture
def reference_raster(tmp_path: Path) -> Path:
    path = tmp_path / "reference.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float64",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
    ) as raster:
        raster.write(np.array([[6.0, 8.0], [2.0, 4.0]]), 1)
    return path


def mean_wind() -> xr.DataArray:
    return xr.DataArray(
        np.full((2, 2), 2.0),
        dims=("y", "x"),
        coords={"y": [0.5, 1.5], "x": [0.5, 1.5]},
    )


def test_prepare_bias_correction_crs(
    reference_raster: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(era5, "retrieve_windspeed_average", lambda *a, **k: mean_wind())
    result = era5.get_data_windspeed_bias_correction(
        SimpleNamespace(module="era5"),
        {},
        {"windspeed_real_average_path": str(reference_raster)},
    )
    np.testing.assert_allclose(result.wnd_bias_correction, [[1, 2], [3, 4]])


def test_select_module_with_wind_means(
    reference_raster: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(era5, "retrieve_windspeed_average", lambda *a, **k: mean_wind())
    result = calculate_windspeed_bias_correction(
        SimpleNamespace(module=["sarah", "era5"]),
        reference_raster,
    )
    np.testing.assert_allclose(result, [[1, 2], [3, 4]])


def test_raster_dimension_order(reference_raster: Path) -> None:
    result = calculate_windspeed_bias_correction(
        None,
        reference_raster,
        data_average=mean_wind().transpose("x", "y"),
        data_crs=4326,
    )
    np.testing.assert_allclose(result.transpose("y", "x"), [[1, 2], [3, 4]])


def test_hourly_average_weights_leap_year(monkeypatch: pytest.MonkeyPatch) -> None:
    def retrieve(**kwargs: object) -> xr.Dataset:
        year = int(kwargs["year"][0])
        time = pd.date_range(
            f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left", tz="UTC"
        )
        speed = 2.0 if year == 2008 else 10.0
        return xr.Dataset(
            {
                "u100": ("time", np.full(len(time), speed), {"units": "m s**-1"}),
                "v100": ("time", np.zeros(len(time))),
            },
            coords={"time": time, "x": [0.5], "y": [0.5]},
        ).rename({"time": "valid_time", "x": "longitude", "y": "latitude"})

    monkeypatch.setattr(era5, "retrieve_data", retrieve)
    result = era5.retrieve_windspeed_average(first_year=2008, last_year=2009)
    assert float(result) == pytest.approx((366 * 2 + 365 * 10) / 731)


@pytest.mark.parametrize("skip_first", [True, False])
def test_prepare_multiple_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    skip_first: bool,
) -> None:
    from atlite import data

    cutout = Cutout(
        tmp_path / "cutout.nc",
        module=["sarah", "era5"],
        x=slice(0, 1),
        y=slice(0, 1),
        time="2013-01-01",
    )
    targets = pd.Series(
        ["height", "roughness"],
        index=pd.MultiIndex.from_tuples(
            [("sarah", "height"), ("era5", "wind")],
            names=["module", "feature"],
        ),
    )
    monkeypatch.setattr(data, "available_features", lambda *a: targets)

    def get_features(
        cutout: Cutout, module: str, *args: object, **kwargs: object
    ) -> xr.Dataset:
        if module == "sarah" and skip_first:
            return xr.Dataset()
        variable, feature = (
            ("height", "height") if module == "sarah" else ("roughness", "wind")
        )
        return xr.Dataset(
            {
                variable: xr.DataArray(
                    np.ones((len(cutout.coords["y"]), len(cutout.coords["x"]))),
                    dims=("y", "x"),
                    coords={k: cutout.coords[k] for k in ["y", "x"]},
                    attrs={"feature": feature, "module": module},
                )
            }
        )

    monkeypatch.setattr(data, "get_features", get_features)
    assert cutout.prepare() is cutout
    assert "roughness" in cutout.data
    expected = {"wind"} if skip_first else {"height", "wind"}
    assert set(np.atleast_1d(cutout.data.attrs["prepared_features"])) == expected
