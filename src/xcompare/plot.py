import xcompare
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings

__all__ = ["plot_panel", "plot_three_panel"]


def plot_panel(
    ax,
    arr,
    longitude="lon",
    latitude="lat",
    cmap=None,
    coastlines=True,
    date_range=None,
    label=None,
    vmin=None,
    vmax=None,
    lon_range=None,
    lat_range=None,
):
    cb = ax.pcolormesh(
        arr[longitude],
        arr[latitude],
        arr,
        shading="auto",
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    if coastlines:
        ax.coastlines(linewidth=0.5)

    lon_range = (-180, 180) if lon_range is None else lon_range
    lat_range = (-90, 90) if lat_range is None else lat_range

    ax.set_extent([*lon_range, *lat_range], ccrs.PlateCarree())

    ax.text(0.01, 1.09, label, ha="left", transform=ax.transAxes, fontsize=10)
    ax.text(0.01, 1.02, date_range, ha="left", transform=ax.transAxes, fontsize=10)
    return cb


def plot_three_panel(
    results,
    var,
    projection=ccrs.PlateCarree(),
    labels=["Dataset 1", "Dataset 2"],
    coastlines=True,
    vmin=None,
    vmax=None,
    diffvmin=None,
    diffvmax=None,
    lon_range=None,
    lat_range=None,
    sigma=1.5,
    cmap=None,
):
    arr1 = results["ds1_orig"][var]
    arr2 = results["ds2_orig"][var]
    arrdiff = results["diff"][var]

    arr1.load()
    arr2.load()
    arrdiff.load()

    # Check to see if area field made it all the way through
    area = results["diff"]["area"] if "area" in results["diff"].variables else None

    # Get time ranges from plots
    daterange1 = (
        results["ds1"].date_range if "date_range" in results["ds1"].attrs else None
    )
    daterange2 = (
        results["ds2"].date_range if "date_range" in results["ds2"].attrs else None
    )

    daterange1 = (
        f"Years {daterange1[0]} - {daterange1[1]}"
        if isinstance(daterange1, tuple)
        else None
    )
    daterange2 = (
        f"Years {daterange2[0]} - {daterange2[1]}"
        if isinstance(daterange2, tuple)
        else None
    )

    if lon_range is not None or lat_range is not None:
        arr1_rgd = results["ds1"][var]
        arr2_rgd = results["ds2"][var]
        diff_rgd = results["diff"][var]

        arr1_rgd.load()
        arr2_rgd.load()
        diff_rgd.load()

        if len(arr1_rgd.lon.shape) == 2:
            if lon_range is not None:
                arr1_rgd = arr1_rgd.where(
                    (arr1_rgd.lon > lon_range[0]) & (arr1_rgd.lon < lon_range[1]),
                    drop=True,
                )
                arr2_rgd = arr2_rgd.where(
                    (arr2_rgd.lon > lon_range[0]) & (arr2_rgd.lon < lon_range[1]),
                    drop=True,
                )
                diff_rgd = arrdiff.where(
                    (diff_rgd.lon > lon_range[0]) & (diff_rgd.lon < lon_range[1]),
                    drop=True,
                )

                if area is not None:
                    area = area.where(
                        (area.lon > lon_range[0]) & (area.lon < lon_range[1]), drop=True
                    )

            if lat_range is not None:
                arr1_rgd = arr1_rgd.where(
                    (arr1_rgd.lat > lat_range[0]) & (arr1_rgd.lat < lat_range[1]),
                    drop=True,
                )
                arr2_rgd = arr2_rgd.where(
                    (arr2_rgd.lat > lat_range[0]) & (arr2_rgd.lat < lat_range[1]),
                    drop=True,
                )
                diff_rgd = diff_rgd.where(
                    (diff_rgd.lat > lat_range[0]) & (diff_rgd.lat < lat_range[1]),
                    drop=True,
                )

                if area is not None:
                    area = area.where(
                        (area.lat > lat_range[0]) & (area.lat < lat_range[1]), drop=True
                    )

        else:
            lon_range = (None, None) if lon_range is None else lon_range
            lat_range = (None, None) if lat_range is None else lat_range

            arr1_rgd = arr1_rgd.sel(lon=slice(*lon_range)).sel(lat=slice(*lat_range))
            arr2_rgd = arr2_rgd.sel(lon=slice(*lon_range)).sel(lat=slice(*lat_range))
            diff_rgd = diff_rgd.sel(lon=slice(*lon_range)).sel(lat=slice(*lat_range))

        try:
            if area is not None:
                area = area.fillna(0.0)
                stats = (
                    xcompare.xr_stats.xr_stats_2d(arr1_rgd, arr2_rgd, area, fmt="dict")
                    if area.sum() > 0.0
                    else None
                )
            else:
                stats = None
            concat = xr.concat([arr1_rgd, arr2_rgd], dim="dset")
        except Exception as e:
            diff_rgd = arrdiff
            stats = None
            concat = xr.concat([arr1, arr2], dim="dset")
            warnings.warn(f"Unable to calculate stats: {e}")
            pass

    else:
        diff_rgd = arrdiff
        concat = xr.concat([arr1, arr2], dim="dset")
        try:
            stats = {
                "bias": arrdiff.bias,
                "rmse": arrdiff.rmse,
                "rsquared": arrdiff.rsquared,
            }
        except Exception as e:
            stats = None
            warnings.warn(f"Unable to extract stats from metadata: {e}")
            pass

    if vmin is None or vmax is None:
        vmin = concat.mean() - sigma * concat.std() if vmin is None else vmin
        vmax = concat.mean() + sigma * concat.std() if vmax is None else vmax

    if diffvmin is None or diffvmax is None:
        val = np.max(
            (
                np.abs(diff_rgd.mean() - sigma * diff_rgd.std()),
                np.abs(diff_rgd.mean() + sigma * diff_rgd.std()),
            )
        )
        diffvmin = -1.0 * val
        diffvmax = val

    if len(labels) == 2:
        labels = labels + ["Difference (A-B)"]

    fig = plt.figure(figsize=(8.5, 11))

    ax1 = plt.subplot(3, 1, 1, projection=projection, facecolor="#c3c3c3")
    cb1 = plot_panel(
        ax1,
        arr1,
        coastlines=coastlines,
        label="a. " + labels[0],
        vmin=vmin,
        vmax=vmax,
        lon_range=lon_range,
        lat_range=lat_range,
        cmap=cmap,
        date_range=daterange1,
    )

    plt.colorbar(cb1, ax=ax1, orientation="vertical")

    ax2 = plt.subplot(3, 1, 2, projection=projection, facecolor="#c3c3c3")
    cb2 = plot_panel(
        ax2,
        arr2,
        coastlines=coastlines,
        label="b. " + labels[1],
        vmin=vmin,
        vmax=vmax,
        lon_range=lon_range,
        lat_range=lat_range,
        cmap=cmap,
        date_range=daterange2,
    )

    plt.colorbar(cb2, ax=ax2, orientation="vertical")

    ax3 = plt.subplot(3, 1, 3, projection=projection, facecolor="#c3c3c3")
    cb3 = plot_panel(
        ax3,
        arrdiff,
        cmap="RdBu_r",
        coastlines=coastlines,
        label="c. " + labels[2],
        lon_range=lon_range,
        lat_range=lat_range,
        vmin=diffvmin,
        vmax=diffvmax,
    )

    plt.colorbar(cb3, ax=ax3, orientation="vertical")

    # if lat_range is None and lon_range is None:
    ax3.text(
        0.01,
        -0.07,
        f"Max = {np.array(diff_rgd.max())}",
        ha="left",
        transform=ax3.transAxes,
    )
    ax3.text(
        0.01,
        -0.14,
        f"Min = {np.array(diff_rgd.min())}",
        ha="left",
        transform=ax3.transAxes,
    )
    if stats is not None:
        ax3.text(
            0.01, -0.21, f"Bias = {stats['bias']}", ha="left", transform=ax3.transAxes
        )

        ax3.text(
            0.01, -0.28, f"RMSE = {stats['rmse']}", ha="left", transform=ax3.transAxes
        )

        ax3.text(
            0.01,
            -0.35,
            f"r^2 = {stats['rsquared']}",
            ha="left",
            transform=ax3.transAxes,
        )

    plt.suptitle(var)

    return fig
