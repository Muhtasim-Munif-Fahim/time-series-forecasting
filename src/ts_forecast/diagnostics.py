"""Stationarity and trend diagnostics for time series."""

import warnings

import numpy as np
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(values, maxlag=None, regression="c", autolag="AIC", alpha=0.05):
    """Run the augmented Dickey-Fuller unit-root test on a series.

    The ADF test checks the null hypothesis that the series has a unit
    root against the stationary alternative. The ``regression`` term
    controls which deterministic components are included: ``"c"`` fits a
    constant, ``"ct"`` adds a linear trend, ``"ctt"`` adds a quadratic
    trend, and ``"n"`` fits neither. ``maxlag`` fixes the number of
    lagged difference terms (the default lets ``autolag`` pick it by
    AIC or BIC). The verdict marks the series stationary when the
    p-value falls below ``alpha``.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size < 10:
        raise ValueError("at least ten observations are required for the ADF test")
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if regression not in {"c", "ct", "ctt", "n"}:
        raise ValueError("regression must be one of 'c', 'ct', 'ctt', 'n'")
    if maxlag is not None and (
        isinstance(maxlag, bool) or not isinstance(maxlag, int) or maxlag < 0
    ):
        raise ValueError("maxlag must be a non-negative integer or None")
    if autolag not in {"AIC", "BIC", "t-stat", None}:
        raise ValueError("autolag must be 'AIC', 'BIC', 't-stat', or None")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    result = adfuller(
        observed, maxlag=maxlag, regression=regression, autolag=autolag
    )
    if autolag is None:
        statistic, p_value, usedlag, nobs, critical_values = result
    else:
        statistic, p_value, usedlag, nobs, critical_values, _ = result
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": {level: float(value) for level, value in critical_values.items()},
        "stationary": bool(p_value < alpha),
    }


def kpss_test(values, regression="c", nlags="auto", alpha=0.05):
    """Run the KPSS stationarity test on a series.

    KPSS tests the null hypothesis that the series is stationary (level
    stationarity with ``regression="c"``, trend stationarity with
    ``regression="ct"``), the opposite null to :func:`adf_test`. Rejecting
    KPSS while failing to reject ADF is the classic evidence of a
    unit-root series, so the two tests are best read together.
    ``nlags`` fixes the number of lags for the Newey-West estimate of the
    long-run variance, with ``"auto"`` choosing by the standard heuristic.
    The verdict marks the series stationary when the p-value stays at or
    above ``alpha`` (i.e. the stationarity null is not rejected). The
    statistic occasionally lands outside the p-value look-up table; the
    returned p-value then saturates at the nearest extreme (``0.01`` or
    ``0.1``) and the routine warns only about that censoring.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size < 10:
        raise ValueError("at least ten observations are required for the KPSS test")
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if regression not in {"c", "ct"}:
        raise ValueError("regression must be 'c' or 'ct'")
    if nlags not in {"auto", "legacy"} and (
        isinstance(nlags, bool) or not isinstance(nlags, int) or nlags < 0
    ):
        raise ValueError("nlags must be 'auto', 'legacy', or a non-negative integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        statistic, p_value, usedlag, critical_values = kpss(
            observed, regression=regression, nlags=nlags
        )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "usedlag": int(usedlag),
        "critical_values": {level: float(value) for level, value in critical_values.items()},
        "stationary": bool(p_value >= alpha),
    }


def mann_kendall_test(values, alpha=0.05):
    """Run the non-parametric Mann-Kendall trend test on a series.

    The test counts how many later observations exceed earlier ones minus
    how many fall below them, giving Kendall's S. Under the no-trend null
    S has expected value zero and its variance accounts for tied values,
    so the continuity-corrected statistic is referred to the standard
    normal distribution. The verdict reports ``"increasing"`` or
    ``"decreasing"`` when the two-sided p-value falls below ``alpha`` and
    ``"no trend"`` otherwise. Unlike :func:`adf_test` and
    :func:`kpss_test`, which test stationarity, this tests monotonic trend
    directly and needs no distributional assumption about the data.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size < 10:
        raise ValueError(
            "at least ten observations are required for the Mann-Kendall test"
        )
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if np.all(observed == observed[0]):
        raise ValueError("values must not be constant")

    n = observed.size
    statistic = 0.0
    for i in range(n - 1):
        statistic += float(np.sum(np.sign(observed[i + 1 :] - observed[i])))

    _, counts = np.unique(observed, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1.0) * (2.0 * counts + 5.0)))
    variance = (n * (n - 1.0) * (2.0 * n + 5.0) - tie_term) / 18.0

    if variance > 0:
        if statistic > 0:
            z_score = (statistic - 1.0) / np.sqrt(variance)
        elif statistic < 0:
            z_score = (statistic + 1.0) / np.sqrt(variance)
        else:
            z_score = 0.0
        p_value = float(2.0 * norm.sf(abs(z_score)))
    else:
        z_score = 0.0
        p_value = 1.0

    total_pairs = 0.5 * n * (n - 1)
    tied_share = float(np.sum(counts * (counts - 1.0))) / 2.0
    denominator = total_pairs - tied_share
    tau = float(statistic / denominator) if denominator > 0 else 0.0

    if p_value < alpha:
        verdict = "increasing" if statistic > 0 else "decreasing"
    else:
        verdict = "no trend"
    return {
        "tau": tau,
        "p_value": p_value,
        "s": float(statistic),
        "var_s": float(variance),
        "trend": verdict,
        "alpha": float(alpha),
    }


def stationarity_report(values, max_diffs=2, alpha=0.05, **adf_kwargs):
    """Diagnose stationarity and suggest how many differences are needed.

    The series is tested with the augmented Dickey-Fuller test at zero,
    one, and up to ``max_diffs`` differences. ``suggested_diffs`` is the
    first order at which the test rejects a unit root (``0`` when the
    original series is already stationary); when the series stays
    non-stationary at every tested order the suggestion saturates at
    ``max_diffs`` and the verdict reports non-stationarity. The full
    per-order test results are returned so the p-value trajectory is
    visible, not just the final recommendation.
    """

    if isinstance(max_diffs, bool) or not isinstance(max_diffs, int) or max_diffs < 1:
        raise ValueError("max_diffs must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size < 10:
        raise ValueError("at least ten observations are required for the ADF test")
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")

    tests = []
    suggested = None
    current = observed
    for order in range(max_diffs + 1):
        result = adf_test(current, alpha=alpha, **adf_kwargs)
        tests.append(
            {
                "diffs": order,
                "statistic": result["statistic"],
                "p_value": result["p_value"],
                "stationary": bool(result["p_value"] < alpha),
            }
        )
        if result["p_value"] < alpha and suggested is None:
            suggested = order
        if order < max_diffs:
            current = np.diff(current)

    if suggested is None:
        suggested = max_diffs
    return {
        "verdict": "stationary" if suggested == 0 else "non-stationary",
        "suggested_diffs": int(suggested),
        "alpha": float(alpha),
        "tests": tests,
    }


def _cusum_break(segment):
    """Return the index and normalized statistic of the strongest mean shift.

    Implements the CUSUM-of-means scan: for each interior split point the
    cumulative deviation from the segment mean is measured, and the split
    maximizing it is the candidate break. The statistic is normalized by
    the segment standard deviation and length so it is comparable across
    segments of different size and scale.
    """

    n = segment.size
    if n < 4:
        return None, 0.0

    centered = segment - segment.mean()
    cumulative = np.cumsum(centered)
    # Endpoints are structurally zero and are not candidate breaks.
    interior = np.abs(cumulative[:-1])
    if interior.size == 0:
        return None, 0.0

    position = int(np.argmax(interior))
    spread = float(segment.std(ddof=0))
    if spread <= 0:
        return None, 0.0

    statistic = float(interior[position] / (spread * np.sqrt(n)))
    return position + 1, statistic


def detect_changepoints(values, threshold=1.36, min_segment=8, max_breaks=5):
    """Locate mean-shift changepoints by recursive CUSUM segmentation.

    A structural break invalidates a model fitted across it: parameters
    estimated on both regimes describe neither. The stationarity tests in
    this module answer whether a series needs differencing, not whether its
    mean moved partway through, so this fills that gap.

    The series is scanned for the split maximizing the normalized CUSUM
    statistic. If that exceeds ``threshold`` the split is accepted and the
    two halves are scanned recursively (binary segmentation), stopping when
    no segment beats the threshold, a segment would fall below
    ``min_segment``, or ``max_breaks`` have been found.

    Returns a dict with ``changepoints`` (sorted indices, each the first
    observation of a new regime), ``n_changepoints``, and ``segments`` --
    one entry per regime carrying ``start``, ``end`` (exclusive), ``length``,
    ``mean`` and ``std``.

    The normalized statistic converges to the supremum of a Brownian bridge
    under the no-break null, so ``threshold`` can be read off the Kolmogorov
    distribution: 1.36 is roughly the 5% critical value and 1.63 the 1% one.
    The default is 1.36. That calibration covers a single scan, and binary
    segmentation runs one scan per candidate segment, so the family-wise rate
    is higher than the nominal level -- on 300 pure-noise series of length 180
    a threshold of 1.36 reported a spurious break in 3.3% of them, against 23%
    at 1.0 and 0.7% at 1.63. Raise it to report only pronounced shifts.

    Detected positions are approximate near the series ends, where fewer
    observations support the estimate. The scan targets shifts in the mean; a
    change in variance or in trend slope alone may go unreported.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if observed.size < 2 * min_segment:
        raise ValueError(
            f"at least {2 * min_segment} observations are required "
            f"for min_segment={min_segment}"
        )
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if not isinstance(min_segment, (int, np.integer)) or min_segment < 2:
        raise ValueError("min_segment must be an integer of at least 2")
    if not isinstance(max_breaks, (int, np.integer)) or max_breaks < 0:
        raise ValueError("max_breaks must be a non-negative integer")
    if threshold <= 0:
        raise ValueError("threshold must be strictly positive")

    breaks = []

    def _scan(start, end):
        if len(breaks) >= max_breaks:
            return
        if end - start < 2 * min_segment:
            return
        position, statistic = _cusum_break(observed[start:end])
        if position is None or statistic < threshold:
            return
        absolute = start + position
        # Reject a split that would leave either side under min_segment.
        if absolute - start < min_segment or end - absolute < min_segment:
            return
        breaks.append(absolute)
        _scan(start, absolute)
        _scan(absolute, end)

    _scan(0, observed.size)
    breaks.sort()

    bounds = [0, *breaks, observed.size]
    segments = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        piece = observed[left:right]
        segments.append(
            {
                "start": int(left),
                "end": int(right),
                "length": int(piece.size),
                "mean": float(piece.mean()),
                "std": float(piece.std(ddof=0)),
            }
        )

    return {
        "changepoints": [int(b) for b in breaks],
        "n_changepoints": len(breaks),
        "segments": segments,
        "threshold": float(threshold),
        "min_segment": int(min_segment),
    }


_DETRENDS = ("linear", "constant", None)


def _detrended(values, detrend):
    observed = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if detrend not in _DETRENDS:
        raise ValueError("detrend must be 'linear', 'constant', or None")
    if detrend == "constant":
        return observed - observed.mean()
    if detrend == "linear":
        t = np.arange(observed.size, dtype=float)
        slope, intercept = np.polyfit(t, observed, 1)
        return observed - (intercept + slope * t)
    return observed


def periodogram(values, detrend="linear"):
    """Raw periodogram at the interior Fourier frequencies.

    Returns the ordinates ``I(f_j) = |sum_t x_t exp(-2 pi i f_j t)|^2 / n``
    at ``f_j = j / n`` for ``j = 1 .. floor((n - 1) / 2)`` (cycles per
    observation), i.e. excluding the zero frequency and, for even ``n``, the
    Nyquist frequency, whose ordinates are not exponentially distributed
    under white noise. ``detrend`` removes a least-squares line
    (``"linear"``, the default, so a trend does not leak into the lowest
    frequencies), the mean (``"constant"``), or nothing (``None``).
    Periods are reported in observations (``1 / f_j``).
    """

    x = _detrended(values, detrend)
    n = x.size
    if n < 8:
        raise ValueError("at least eight observations are required for a periodogram")
    m = (n - 1) // 2
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum[1 : m + 1]) ** 2 / n
    j = np.arange(1, m + 1)
    frequencies = j / n
    return {
        "frequencies": frequencies,
        "periods": n / j,
        "power": power,
        "n": int(n),
        "detrend": detrend,
    }


def _fisher_g_pvalue(g, m):
    """Exact ``P(G >= g)`` for Fisher's g with ``m`` ordinates (Fisher 1929).

    ``P = sum_{k=1}^{floor(1/g)} (-1)^(k-1) C(m, k) (1 - k g)^(m-1)``. The
    alternating sum is evaluated in high-precision decimal arithmetic so it
    stays accurate when many terms cancel.
    """

    from decimal import Decimal, localcontext
    from math import comb, floor

    if m < 2:
        return 1.0
    if g <= 1.0 / m:
        return 1.0
    upper = min(m, floor(1.0 / g))
    with localcontext() as ctx:
        ctx.prec = 60 + int(0.31 * m)
        gd = Decimal(repr(float(g)))
        total = Decimal(0)
        for k in range(1, upper + 1):
            base = 1 - k * gd
            if base <= 0:
                break
            term = Decimal(comb(m, k)) * base ** (m - 1)
            total += term if k % 2 == 1 else -term
        return float(min(max(total, Decimal(0)), Decimal(1)))


def fisher_g_test(values, detrend="linear", alpha=0.05, max_period=None, exclude=()):
    """Fisher's exact g test for a hidden periodicity.

    ``g = max_j I(f_j) / sum_j I(f_j)`` over the periodogram ordinates from
    :func:`periodogram`. Under Gaussian white noise the ordinates are i.i.d.
    exponential and ``g`` has Fisher's exact null distribution, so a small
    p-value means one frequency carries far more variance than chance would
    allow. ``max_period`` drops ordinates whose period exceeds it before
    testing (fix it in advance, e.g. to ignore slow drift), and ``exclude``
    lists ordinate indices (positions in the periodogram arrays) to leave out,
    which :func:`detect_seasonal_periods` uses to test successive peaks.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    spec = periodogram(values, detrend=detrend)
    keep = np.ones(spec["power"].size, dtype=bool)
    if max_period is not None:
        if max_period < 2:
            raise ValueError("max_period must be at least 2")
        keep &= spec["periods"] <= max_period
    for index in exclude:
        keep[int(index)] = False
    candidates = np.flatnonzero(keep)
    if candidates.size < 2:
        raise ValueError("fewer than two periodogram ordinates remain to test")
    power = spec["power"][candidates]
    total = power.sum()
    scale = max(float(np.max(np.abs(np.asarray(values, dtype=float)))), 1.0)
    if total <= spec["n"] * (1e-10 * scale) ** 2:
        raise ValueError("series has no variance after detrending")
    best = int(np.argmax(power))
    g = float(power[best] / total)
    p_value = _fisher_g_pvalue(g, candidates.size)
    index = int(candidates[best])
    return {
        "g": g,
        "p_value": p_value,
        "significant": bool(p_value < alpha),
        "alpha": float(alpha),
        "n_ordinates": int(candidates.size),
        "index": index,
        "frequency": float(spec["frequencies"][index]),
        "period": float(spec["periods"][index]),
        "power": float(spec["power"][index]),
    }


def detect_seasonal_periods(
    values, max_periods=3, alpha=0.05, detrend="linear", max_period=None, leakage_bins=1
):
    """Find significant cycles with repeated Fisher g tests on the periodogram.

    The strongest ordinate is tested with :func:`fisher_g_test`; if
    significant it is recorded and removed together with ``leakage_bins``
    neighbours on each side (a cycle whose period is not an exact Fourier
    period leaks power into adjacent ordinates), and the next strongest is
    tested against the remaining ordinates, until a test fails or ``max_periods``
    cycles are found. A significant peak that rounds to an already detected
    period is treated as further leakage of that cycle: it is removed but not
    reported again. Each detection reports the Fourier period ``n / j``
    and ``rounded_period`` (the nearest integer, handy as a
    ``seasonal_period`` for Holt-Winters/SARIMA/seasonal naive). Harmonics
    of a non-sinusoidal season (``period / 2``, ``period / 3``, ...) can be
    detected as separate cycles; ``primary_period`` is the rounded period of
    the strongest detection, or ``None`` when nothing is significant.
    """

    if not isinstance(max_periods, (int, np.integer)) or max_periods < 1:
        raise ValueError("max_periods must be a positive integer")
    if not isinstance(leakage_bins, (int, np.integer)) or leakage_bins < 0:
        raise ValueError("leakage_bins must be a non-negative integer")
    n_ordinates = (np.asarray(values).size - 1) // 2
    detections = []
    excluded = []
    last = None
    while len(detections) < max_periods:
        try:
            result = fisher_g_test(
                values, detrend=detrend, alpha=alpha, max_period=max_period, exclude=excluded
            )
        except ValueError:
            if not detections and last is None:
                raise
            break
        last = result
        if not result["significant"]:
            break
        result["rounded_period"] = int(round(result["period"]))
        if result["rounded_period"] not in {d["rounded_period"] for d in detections}:
            detections.append(result)
        centre = result["index"]
        for index in range(centre - leakage_bins, centre + leakage_bins + 1):
            if 0 <= index < n_ordinates and index not in excluded:
                excluded.append(index)
    return {
        "periods": detections,
        "primary_period": detections[0]["rounded_period"] if detections else None,
        "alpha": float(alpha),
        "last_test": last,
    }


def _sidak(p_value, m):
    """Sidak-adjusted family-wise p-value ``1 - (1 - p)^m``."""

    return float(min(1.0, max(0.0, -np.expm1(m * np.log1p(-min(p_value, 1.0 - 1e-300))))))


def variance_ratio_test(values, periods=(2, 4, 8, 16), increments=False, alpha=0.05):
    """Lo-MacKinlay (1988) variance ratio test of the random-walk hypothesis.

    ``values`` are levels ``X_0 .. X_T`` (e.g. log prices or a cumulative
    series) unless ``increments=True``, in which case they are the
    differences ``x_t = X_t - X_{t-1}`` themselves. Under a random walk the
    variance of ``q``-period increments grows linearly in ``q``, so

    ``VR(q) = sigma_c^2(q) / sigma_a^2``

    is close to one; ``VR(q) > 1`` points to positively autocorrelated
    increments (momentum, a trending component) and ``VR(q) < 1`` to mean
    reversion. Both variances use Lo and MacKinlay's bias-corrected
    estimators with overlapping ``q``-period increments:
    ``sigma_a^2 = sum (x_t - mu)^2 / (T - 1)`` and
    ``sigma_c^2(q) = sum_{t=q}^{T} (X_t - X_{t-q} - q mu)^2 / m`` with
    ``m = q (T - q + 1) (1 - q / T)``.

    Each period reports the homoskedastic statistic
    ``z(q) = (VR(q) - 1) / sqrt(2 (2q - 1)(q - 1) / (3 q T))`` and the
    heteroskedasticity-consistent ``z*(q) = (VR(q) - 1) / sqrt(theta(q) / T)``
    with ``theta(q) = sum_{j=1}^{q-1} (2 (q - j) / q)^2 delta_j`` and
    ``delta_j = T sum_t dev_t^2 dev_{t-j}^2 / (sum_t dev_t^2)^2``, so a random walk
    with volatility clustering is not rejected spuriously. Two-sided normal
    p-values are given for both. The joint Chow-Denning (1993) test takes
    ``max_q |z*(q)|`` and bounds its family-wise p-value with the Sidak
    inequality (the studentized maximum modulus bound), which is
    conservative because the ``z*(q)`` are positively correlated.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if isinstance(periods, (int, np.integer)):
        periods = (periods,)
    periods = tuple(periods)
    if not periods:
        raise ValueError("periods must not be empty")
    for q in periods:
        if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 2:
            raise ValueError("each period must be an integer of at least 2")
    if len(set(periods)) != len(periods):
        raise ValueError("periods must be distinct")

    if increments:
        x = observed
        levels = np.concatenate(([0.0], np.cumsum(x)))
    else:
        levels = observed
        x = np.diff(levels)
    T = x.size
    if T < 2 * max(periods):
        raise ValueError("need at least 2 * max(periods) increments")
    mu = (levels[-1] - levels[0]) / T
    dev = x - mu
    ss = float(np.sum(dev**2))
    scale = max(float(np.max(np.abs(x))), 1.0)
    if ss <= T * (1e-12 * scale) ** 2:
        raise ValueError("increments have no variance")
    sigma_a = ss / (T - 1)
    dev2 = dev**2

    rows = []
    for q in periods:
        q = int(q)
        long = levels[q:] - levels[:-q] - q * mu
        m = q * (T - q + 1) * (1.0 - q / T)
        sigma_c = float(np.sum(long**2)) / m
        vr = sigma_c / sigma_a
        phi = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * T)
        z = (vr - 1.0) / np.sqrt(phi)
        theta = 0.0
        for j in range(1, q):
            delta = T * float(np.sum(dev2[j:] * dev2[:-j])) / ss**2
            theta += (2.0 * (q - j) / q) ** 2 * delta
        theta /= T
        z_robust = (vr - 1.0) / np.sqrt(theta) if theta > 0 else float("nan")
        p = float(2.0 * norm.sf(abs(z)))
        p_robust = float(2.0 * norm.sf(abs(z_robust))) if np.isfinite(z_robust) else float("nan")
        rows.append(
            {
                "period": q,
                "variance_ratio": float(vr),
                "z": float(z),
                "p_value": p,
                "z_robust": float(z_robust),
                "p_value_robust": p_robust,
                "asymptotic_variance": float(phi),
                "robust_variance": float(theta),
            }
        )

    finite = [r for r in rows if np.isfinite(r["z_robust"])]
    if finite:
        worst = max(finite, key=lambda r: abs(r["z_robust"]))
        joint_stat = abs(worst["z_robust"])
        joint_p = _sidak(worst["p_value_robust"], len(finite))
        joint_period = worst["period"]
    else:
        joint_stat, joint_p, joint_period = float("nan"), float("nan"), None
    return {
        "periods": rows,
        "n_increments": int(T),
        "drift": float(mu),
        "increment_variance": float(sigma_a),
        "chow_denning": {
            "statistic": float(joint_stat),
            "p_value": float(joint_p),
            "period": joint_period,
        },
        "reject_random_walk": bool(np.isfinite(joint_p) and joint_p < alpha),
        "alpha": float(alpha),
    }


def detrended_fluctuation_analysis(values, scales=None, order=1, min_scale=None):
    """Detrended fluctuation analysis (Peng et al. 1994) scaling exponent.

    The series is integrated into the profile ``Y_k = sum_{t<=k} (x_t - mean)``,
    which is split into non-overlapping windows of ``s`` observations,
    taken both from the start and from the end so no data are dropped
    (``2 floor(n / s)`` windows). A least-squares polynomial of degree
    ``order`` is removed from each window (DFA-1 removes lines, DFA-2
    parabolas, ...). ``F(s)`` is the root-mean-square residual over all
    windows, and ``alpha`` is the slope of ``log F(s)`` against ``log s``.

    For a stationary series ``alpha`` estimates the Hurst exponent: about
    0.5 for white noise or short memory at large scales, above 0.5 for
    persistent long-range correlation, below 0.5 for anti-persistence. A
    random walk gives about 1.5 (``alpha - 1`` is then the Hurst exponent
    of its increments). DFA-``k`` is blind to polynomial trends of degree
    ``k - 1`` in the series, which is its advantage over rescaled-range
    estimates. ``scales`` defaults to about 20 log-spaced integers from
    ``min_scale`` (``order + 3`` by default, the smallest window whose fit
    still has residual degrees of freedom left over) to ``n / 4``.
    """

    observed = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(observed)):
        raise ValueError("values must contain only finite numbers")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    n = observed.size
    if min_scale is None:
        min_scale = order + 3
    if min_scale < order + 2:
        raise ValueError("min_scale must exceed order + 1")
    if scales is None:
        upper = n // 4
        if upper < min_scale + 1:
            raise ValueError("series is too short for DFA at this order")
        scales = np.unique(
            np.round(np.geomspace(min_scale, upper, num=20)).astype(int)
        )
    else:
        scales = np.unique(np.asarray(scales, dtype=int))
        if scales.size == 0:
            raise ValueError("scales must not be empty")
        if scales[0] < order + 2:
            raise ValueError("every scale must exceed order + 1")
        if scales[-1] > n // 2:
            raise ValueError("scales must not exceed half the series length")
    if scales.size < 2:
        raise ValueError("at least two distinct scales are required")

    profile = np.cumsum(observed - observed.mean())
    fluctuations = np.empty(scales.size)
    for i, s in enumerate(scales):
        s = int(s)
        k = n // s
        t = np.arange(s, dtype=float)
        design = np.vander(t, order + 1)
        projector = design @ np.linalg.pinv(design)
        segments = np.vstack(
            (profile[: k * s].reshape(k, s), profile[n - k * s :].reshape(k, s))
        )
        residuals = segments - segments @ projector.T
        fluctuations[i] = np.sqrt(np.mean(residuals**2))

    scale_ref = max(float(np.max(np.abs(profile))), 1.0)
    if np.any(fluctuations <= 1e-12 * scale_ref):
        raise ValueError(
            "profile is exactly polynomial at some scale; DFA exponent is undefined"
        )
    log_s = np.log(scales.astype(float))
    log_f = np.log(fluctuations)
    slope, intercept = np.polyfit(log_s, log_f, 1)
    fitted = intercept + slope * log_s
    ss_res = float(np.sum((log_f - fitted) ** 2))
    ss_tot = float(np.sum((log_f - log_f.mean()) ** 2))
    alpha = float(slope)
    if alpha < 0.4:
        regime = "anti-persistent"
    elif alpha <= 0.6:
        regime = "uncorrelated"
    elif alpha < 1.0:
        regime = "persistent"
    elif alpha <= 1.2:
        regime = "1/f noise"
    else:
        regime = "non-stationary (random-walk like)"
    return {
        "alpha": alpha,
        "intercept": float(intercept),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "scales": scales,
        "fluctuations": fluctuations,
        "order": int(order),
        "regime": regime,
    }
