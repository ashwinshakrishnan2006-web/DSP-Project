"""
CrimeSense – Utility: Haversine Great-Circle Distance
"""
import math


EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in kilometres between two points on
    the Earth specified by (latitude, longitude) in decimal degrees.

    Formula
    -------
    d = 2R · arcsin( sqrt( sin²(Δφ/2) + cos φ₁ · cos φ₂ · sin²(Δλ/2) ) )

    Parameters
    ----------
    lat1, lon1 : float
        Coordinates of the first point in decimal degrees.
    lat2, lon2 : float
        Coordinates of the second point in decimal degrees.

    Returns
    -------
    float
        Distance in kilometres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def haversine_vectorized(lat1, lon1, lat2_arr, lon2_arr):
    """
    Vectorized haversine: compute distances from one point to an array of points.

    Parameters
    ----------
    lat1, lon1 : float
        Origin point.
    lat2_arr, lon2_arr : array-like
        Arrays of destination latitudes and longitudes.

    Returns
    -------
    numpy.ndarray
        Distances in kilometres.
    """
    import numpy as np

    lat1_r = math.radians(lat1)
    lat2_r = np.radians(lat2_arr)
    delta_phi = lat2_r - lat1_r
    delta_lambda = np.radians(lon2_arr) - math.radians(lon1)

    a = (
        np.sin(delta_phi / 2) ** 2
        + math.cos(lat1_r) * np.cos(lat2_r) * np.sin(delta_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
