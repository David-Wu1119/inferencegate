"""Release gate for an inference-serving change: PASS, BLOCK, or INCONCLUSIVE.

A serving change (engine version, model, GPU, quantization, kernel, decoding
method, config) is qualified by replaying the same workload against the
baseline and the candidate several times and asking one question per metric:

    is the candidate meaningfully worse than the baseline?

Two properties separate this from a naive A/B comparison, and both come from
this project's own data (``variance_probe_delta_summary_bothn5_20260623.csv``,
where one of five identical reruns reverses the sign of the headline metric):

1.  **PASS must be earned, not defaulted to.**  A gate that ships whenever it
    fails to prove a regression ships every regression it is too noisy to see.
    PASS here means the confidence interval excludes a harm larger than the
    pre-registered margin, i.e. non-inferiority was demonstrated.

2.  **INCONCLUSIVE is a first-class verdict.**  When the interval straddles the
    margin the honest answer is that the measurement cannot tell, and the
    remedy is more runs, not a coin flip.  ``runs_needed`` says how many.

The margin is the smallest regression that matters operationally and must be
chosen before the runs, not after seeing them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist, mean, stdev

PASS = "PASS"
BLOCK = "BLOCK"
INCONCLUSIVE = "INCONCLUSIVE"
# ERROR is an orchestration outcome, not a decision: a measurement did not
# complete. It outranks BLOCK so a broken adapter can never read as "the gate
# said no", which is a policy decision nobody made.
ERROR = "ERROR"
_RANK = {PASS: 0, INCONCLUSIVE: 1, BLOCK: 2, ERROR: 3}


def combine(verdicts) -> str:
    """One verdict from many: the worst of them.

    The single contract every layer uses, whether the parts are metrics,
    task families or change axes: ERROR outranks BLOCK, BLOCK outranks
    INCONCLUSIVE, INCONCLUSIVE outranks PASS. A set passes only when every
    part passed. An empty set measured nothing and is INCONCLUSIVE, never a
    vacuous PASS.
    """
    verdicts = list(verdicts)
    if not verdicts:
        return INCONCLUSIVE
    unknown = sorted({v for v in verdicts if v not in _RANK})
    if unknown:
        raise ValueError(f"unknown verdict(s) {unknown}")
    return max(verdicts, key=_RANK.__getitem__)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), by Lentz's continued fraction.

    The one special function the gate needs, so it is written out rather
    than pulled from a package: the whole gate runs on a stock interpreter.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 20000):
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / (c if abs(c) > tiny else tiny)
        h *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / (c if abs(c) > tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return front * h / a


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    lp, lq = math.log(p), math.log1p(-p)
    return sum(
        math.exp(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * lp
            + (n - i) * lq
        )
        for i in range(0, k + 1)
    )


def binomial_upper(k: int, n: int, alpha: float) -> float:
    """Exact one-sided (1 - alpha) upper bound on a rate seen k times in n.

    Clopper-Pearson: the largest rate at which observing k or fewer events
    in n independent trials is still at least alpha-probable. With k = 0
    this is 1 - alpha^(1/n); with k > 0 it is larger, which is the whole
    point: one failed session in 31 leaves 0.144 open, not 0.092.
    """
    if n < 1:
        raise ValueError("need at least one trial")
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def sessions_needed_for(k: int, s: float, alpha: float, start: int, cap: int = 5000):
    """Smallest session count from ``start`` that bounds the rate under ``s``,
    if no further session fails; None when no count up to ``cap`` does.

    The bound falls monotonically in the session count for a fixed event
    count, so this is a bisection, not a walk: a uniformly harmful change
    makes every session an event, and walking 5000 counts with an exact
    binomial bound at each cost seconds per decision.
    """
    lo, hi = max(start, 1), cap
    if binomial_upper(k, hi, alpha) > s:
        return None
    if binomial_upper(k, lo, alpha) <= s:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if binomial_upper(k, mid, alpha) <= s:
            hi = mid
        else:
            lo = mid
    return hi


def t_cdf(t: float, df: float) -> float:
    if t < 0:
        return 1.0 - t_cdf(-t, df)
    x = df / (df + t * t)
    return 1.0 - 0.5 * _betainc(df / 2.0, 0.5, x)


def t_quantile(q: float, df: float) -> float:
    """Student t quantile at probability q, any degrees of freedom.

    Bisection on the CDF: a few hundred evaluations, exact to 1e-9, and
    good at the Bonferroni-sized tails the family decisions use, where a
    table indexed by alpha=0.05 was no use at all.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1")
    if df < 1:
        raise ValueError("need at least 2 units to estimate spread")
    if df > 10000:
        return NormalDist().inv_cdf(q)
    if q < 0.5:
        return -t_quantile(1.0 - q, df)
    lo, hi = 0.0, 2.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return (lo + hi) / 2.0


def _t_crit(df: int) -> float:
    if df < 1:
        raise ValueError("need at least 2 runs to estimate spread")
    return t_quantile(0.975, df)


@dataclass
class Decision:
    verdict: str
    metric: str
    n_runs: int  # independent units behind the interval: runs, prompts, or items
    mean_delta: float
    ci_lo: float
    ci_hi: float
    margin: float
    reason: str
    runs_needed: int | None = None
    # What the interval is made of, reported separately so a reader can see
    # that 1000 observations of 200 prompts are 200 prompts' worth of
    # evidence about a prompt population.
    unit: str = "run"
    unique_prompts: int | None = None
    repeats: int | None = None
    observations: int | None = None
    alpha: float = 0.05
    estimator: str = "paired-t"
    seed: int | None = None
    components: dict | None = None
    limited_by: str | None = None
    # Session risk. A session is a failure event when its own mean delta is
    # past the margin. session_failures_observed counts them;
    # session_rate_upper is the exact one-sided (1 - alpha) upper bound on
    # the per-session failure rate given that count, which with zero events
    # is 1 - alpha^(1/R) and with more is larger. Gated only when a
    # session-failure margin is declared, in which case sessions_needed says
    # what R would bound the rate under it if no further session fails.
    session_failures_observed: int | None = None
    session_rate_upper: float | None = None
    sessions_needed: int | None = None
    # Lower bound on the MEAN quality delta over the session distribution,
    # under a declared session-failure margin. The observed-session interval
    # (ci_lo above) only sees the sessions that were sampled; a rare severe
    # session that did not land still lowers the true mean. This mixes the
    # observed-session bound with the worst-case contribution of the
    # unobserved failure tail: (1 - q_upper) * ci_lo + q_upper * (-1). A mean
    # claim PASSes only when this clears the margin, which needs enough
    # sessions to bound the failure rate tightly, not just the frequency
    # minimum. Closes the round-4 mixed-harm fixture; not a coverage
    # theorem, and not exercised by the calibration tables. 2026-09-09.
    session_mean_lb: float | None = None

    def __str__(self) -> str:
        extra = ""
        if self.runs_needed:
            extra = f"  (needs ~{self.runs_needed} runs)"
        if self.unit == "run":
            size = f"n={self.n_runs} runs"
        elif self.unit == "prompt":
            size = f"n={self.unique_prompts} prompts x {self.repeats}"
        else:
            size = f"n={self.n_runs} items"
        level = f"CI{100 * (1 - self.alpha):.4g}%"
        return (
            f"{self.verdict:13} {self.metric:26} {size} "
            f"delta={self.mean_delta:+7.3f} "
            f"{level}=[{self.ci_lo:+7.3f},{self.ci_hi:+7.3f}] "
            f"margin={self.margin:g}{extra}\n"
            f"{'':13} {self.reason}"
        )


def decide(
    deltas: list[float],
    margin: float,
    metric: str = "metric",
    higher_is_better: bool = True,
) -> Decision:
    """Qualify one metric from repeated per-run deltas (candidate - baseline).

    Args:
        deltas: One delta per replayed run. At least two runs are required;
            a single run can never produce PASS, because it carries no
            estimate of its own noise.
        margin: Smallest regression that matters, as a positive number in the
            metric's units. Pre-register it.
        metric: Name used in the report.
        higher_is_better: False for latency-like metrics, where a positive
            delta is a regression.

    Returns:
        A :class:`Decision`. PASS means non-inferiority was demonstrated,
        BLOCK means a regression larger than *margin* was demonstrated, and
        INCONCLUSIVE means the interval straddles the margin.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    n = len(deltas)
    bad = sum(1 for d in deltas if not math.isfinite(d))
    if bad:
        # A run whose metric could not be computed (no latencies observed, no
        # output tokens) is a measurement that did not happen. It is ERROR,
        # never a verdict: a NaN that fell through to stdev crashed the whole
        # qualification with exit 1, which a shell reads as BLOCK.
        return Decision(
            ERROR,
            metric,
            n,
            float("nan"),
            float("nan"),
            float("nan"),
            margin,
            f"{bad} of {n} run deltas are not finite, so {metric} was not "
            f"measured in every session; fix the measurement before reading "
            f"a verdict",
            repeats=n,
            observations=n,
        )
    if n < 2:
        return Decision(
            INCONCLUSIVE,
            metric,
            n,
            deltas[0] if n else float("nan"),
            float("-inf"),
            float("inf"),
            margin,
            "a single run carries no estimate of its own noise, so it can "
            "neither demonstrate nor rule out a regression",
            runs_needed=5,
        )

    # Orient so that positive always means better.
    oriented = [d if higher_is_better else -d for d in deltas]
    m = mean(oriented)
    sd = stdev(oriented)
    half = _t_crit(n - 1) * sd / math.sqrt(n)
    lo, hi = m - half, m + half

    if hi < -margin:
        verdict = BLOCK
        reason = (
            f"the whole interval is worse than the -{margin:g} margin, so a "
            f"regression that matters is demonstrated"
        )
        needed = None
    elif lo > -margin:
        verdict = PASS
        if lo > margin:
            reason = (
                f"the whole interval is better than +{margin:g}: not merely "
                f"non-inferior but a demonstrated improvement"
            )
        else:
            reason = (
                f"the whole interval is better than the -{margin:g} margin, "
                f"so non-inferiority is demonstrated"
            )
        needed = None
    else:
        verdict = INCONCLUSIVE
        needed = runs_needed(sd, margin, m)
        reason = (
            f"the interval straddles the -{margin:g} margin, so these runs "
            f"cannot separate a real regression from run-to-run noise"
        )

    # Report in the caller's original orientation.
    if not higher_is_better:
        m, lo, hi = -m, -hi, -lo
    return Decision(
        verdict, metric, n, m, lo, hi, margin, reason, needed, repeats=n, observations=n
    )


def runs_needed(sd: float, margin: float, mean_delta: float, cap: int = 500) -> int:
    """Smallest n that would move the interval clear of the margin.

    Answers the only useful follow-up to INCONCLUSIVE. The binding constraint
    is not the interval's width on its own but its width relative to the gap
    between the observed mean and the ``-margin`` boundary: a result sitting
    almost exactly on the margin needs many runs to resolve, however tight it
    already looks. Both arguments are in the oriented scale where positive is
    better, so the boundary sits at ``-margin``.

    Returns ``cap`` when the mean lands on the boundary, where no feasible
    number of runs settles it and the margin itself has to be revisited.
    """
    gap = abs(mean_delta + margin)
    if gap <= 0 or sd <= 0:
        return cap if gap <= 0 else 2
    n = 2
    while n < cap:
        if _t_crit(n - 1) * sd / math.sqrt(n) < gap:
            return n
        n += 1
    return cap


def decide_paired_grid(
    pairs: dict[str, list[tuple[bool, bool]]],
    margin: float,
    metric: str = "success_rate",
    alpha: float = 0.05,
    unit: str = "prompt",
    session_failure_margin: float | None = None,
    diagnostic: bool = False,
    mean_quality_claim: bool = False,
) -> Decision:
    """Qualify a pass/fail metric measured on the same prompts in repeated sessions.

    ``pairs`` maps a prompt id to its (baseline, candidate) outcome in each
    session, in session order, every prompt with the same number of sessions.
    The claim is about the population of prompts the workload was drawn from,
    so the unit of evidence is the prompt, and replaying a prompt again is
    another look at the same prompt rather than another prompt.

    Two-way crossed random-effects model, ``D[p][r] = mu + a_p + s_r + e_pr``:
    prompt effects, session effects, and residual noise. The variance of the
    grand mean is ``sigma_p^2/P + sigma_s^2/R + sigma_e^2/(PR)``; it is
    estimated from the between-prompt, between-session and interaction mean
    squares, floored at the prompt-only term so a noisy session estimate can
    never buy precision, and turned into a t-interval with Satterthwaite
    degrees of freedom. When one session moves every prompt together, the
    session term dominates, the degrees of freedom fall toward R-1 and the
    interval widens: five sessions cannot tell such a shock from the change.
    See docs/gate_repair_2026-09/METHOD_NOTE.md.

    Args:
        pairs: Per-prompt, per-session paired outcomes.
        margin: Smallest drop in rate that matters, in points of proportion.
        alpha: Two-sided error rate for the interval; the families are gated
            at 0.05 divided by their number.
        unit: What a key of ``pairs`` is, for the report: "prompt" or "item".
        session_failure_margin: If given, the largest per-session probability
            of a failure event the claim must exclude, where a session is a
            failure event when its own mean delta is past ``margin``. The
            event count is bounded with the exact one-sided binomial bound at
            this decision's alpha: with zero events in R sessions that is
            1 - alpha^(1/R) (0.45 at R=5, 0.098 at R=29), and with one event
            in 31 it is 0.144. A PASS requires the bound to lie under the
            margin; otherwise the decision is INCONCLUSIVE and names the
            session count that would bound it if no further session fails.
            Round-1 and round-2 review findings (2026-09-05): five sessions
            passed a change that fails whole sessions 15% of the time 44% of
            the time, and a zero-occurrence formula passed one observed
            failure in 31. No number of prompts substitutes for sessions.
        diagnostic: Observed-sessions analysis that cannot approve: a
            would-be PASS is reported as INCONCLUSIVE with its diagnostic
            result in the reason. A release gate uses one of the two; the
            bare call (neither) is for analysis code and says so.
        mean_quality_claim: Apply the session-mean lower bound (the round-4
            unobserved-tail fix). Set for answer quality, where a whole
            session can be catastrophically wrong without any other metric
            showing it, so an unobserved severe session biases the mean.
            Not set for request success or schema compliance, whose own
            metric directly measures the session-failure mode the frequency
            guard already bounds.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha}")
    if session_failure_margin is not None and not 0.0 < session_failure_margin < 1.0:
        raise ValueError("session_failure_margin must lie strictly between 0 and 1")
    P = len(pairs)
    if P == 0:
        return Decision(
            INCONCLUSIVE,
            metric,
            0,
            float("nan"),
            float("-inf"),
            float("inf"),
            margin,
            "no paired prompts, so nothing was measured",
            unit=unit,
            unique_prompts=0,
            repeats=0,
            observations=0,
            alpha=alpha,
            estimator="crossed-t",
        )
    lengths = {len(v) for v in pairs.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(
            f"every prompt must have the same number of sessions, got {sorted(lengths)}"
        )
    R = lengths.pop()
    ids = sorted(pairs)
    D = [[float(c) - float(b) for b, c in pairs[p]] for p in ids]
    observations = P * R
    base_rate = sum(b for p in ids for b, _ in pairs[p]) / observations
    dbar = [sum(row) / R for row in D]
    grand = sum(dbar) / P
    m_r = [sum(D[i][r] for i in range(P)) / P for r in range(R)]
    failures = sum(1 for x in m_r if x < -margin)
    q_upper = binomial_upper(failures, R, alpha)
    needed = None
    gated_out = session_failure_margin is not None and q_upper > session_failure_margin
    if gated_out:
        needed = sessions_needed_for(failures, session_failure_margin, alpha, R)
    common = dict(
        unit=unit,
        unique_prompts=P,
        repeats=R,
        observations=observations,
        alpha=alpha,
        estimator="crossed-t" if R > 1 else "paired-t",
        session_failures_observed=failures,
        session_rate_upper=q_upper,
        sessions_needed=needed,
    )
    if P < 2:
        return Decision(
            INCONCLUSIVE,
            metric,
            P,
            grand,
            float("-inf"),
            float("inf"),
            margin,
            f"one {unit} carries no estimate of its own spread, so it can "
            f"neither demonstrate nor rule out a regression",
            **common,
        )
    if base_rate == 0:
        # Nothing succeeded in the baseline, so there is no height to fall
        # from and the interval collapses onto zero. Reporting that as
        # non-inferiority would pass a change on a metric that measured
        # nothing, which is usually a broken grader rather than a healthy
        # release.
        return Decision(
            INCONCLUSIVE,
            metric,
            P,
            grand,
            0.0,
            0.0,
            margin,
            f"{metric} never succeeded in the baseline, so this metric "
            f"cannot evidence a regression; check the grader before "
            f"trusting the verdict",
            **common,
        )

    prompt_term = sum((x - grand) ** 2 for x in dbar) / (P - 1) / P
    session_term = None
    v_r = 0.0
    if R > 1:
        v_r = sum((x - grand) ** 2 for x in m_r) / (R - 1)
        v_e = sum(
            (D[i][r] - dbar[i] - m_r[r] + grand) ** 2
            for i in range(P)
            for r in range(R)
        ) / ((P - 1) * (R - 1))
        session_term = max(0.0, v_r / R - v_e / (P * R))
    se2 = prompt_term + (session_term or 0.0)
    components = {"prompt": prompt_term, "session": session_term}
    limited_by = "sessions" if (session_term or 0.0) > prompt_term else "prompts"
    common.update(components=components, limited_by=limited_by)

    if session_term:
        df = se2**2 / (prompt_term**2 / (P - 1) + (v_r / R) ** 2 / (R - 1))
        df = max(1.0, df)
    else:
        df = float(P - 1)
    # The half-width is floored at the rule-of-three bound for P prompts.
    # With no observed variation the t-interval collapses to a point, and a
    # zero-width interval satisfies non-inferiority at any margin, letting
    # two agreeing prompts "demonstrate" what a thousand would. With a little
    # variation from a handful of prompts it is nearly as bad: four prompts
    # carrying only flake noise give a narrow interval around zero whatever
    # the population does, and the 2026-09-05 dev calibration measured 73%
    # family coverage at a nominal 98.75% for exactly that shape, plus three
    # false PASSes from ten-prompt families that sampled no broken prompt.
    # The bound says what P prompts can resolve at all: with no contrary
    # observation among P, a difference up to -ln(alpha)/P stays consistent
    # with the data. P counts prompts; replaying them does not raise it.
    bound = -math.log(alpha) / P
    spread = t_quantile(1.0 - alpha / 2.0, df) * math.sqrt(se2)
    floored = bound > spread
    half = max(spread, bound)
    lo, hi = grand - half, grand + half

    # The mean-quality claim under a session policy. The observed-session
    # interval [lo, hi] only reflects the sessions that were sampled; a rare
    # severe session (a whole-session failure) that did not land still lowers
    # the true mean over the session distribution. So a mean PASS is not the
    # observed interval clearing the margin -- it is the observed bound mixed
    # with the worst that the unobserved failure tail could contribute. The
    # failure fraction is at most q_upper (the count-aware binomial), and a
    # failure session's mean is at worst SESSION_FLOOR = -1 for a paired
    # {-1,0,+1} delta, so the true mean is at least
    #     (1 - q_upper) * lo + q_upper * (-1).
    # Round-4 review (2026-09-09): a candidate 7 points persistently worse
    # plus a 4% whole-session outage is -0.107 on average, but 30.6% of
    # 29-session runs never sampled the outage and the earlier gate certified
    # the observed -0.07. This closes that: with lo=-0.07 and q_upper=0.098
    # the bound is -0.161, which the -0.10 margin does not cover. When failures
    # are observed they are already in lo, and counting them again at the floor
    # only lowers the bound further. What is demonstrated: this closes the
    # mixed-harm fixture (false PASS 0.306 -> 0 at 29 sessions). What is NOT
    # demonstrated (2026-09-09 review): a coverage theorem. lo is the crossed
    # prompt/session interval with a prompt-count floor, not an established
    # lower confidence bound for the conditional mean of non-failure sessions,
    # and the joint coverage of the interval and the failure-rate event is not
    # justified here. Treat this as regression evidence, not certification.
    # The frequency guard (gated_out) stays a separate necessary condition;
    # this adds the mean.
    SESSION_FLOOR = -1.0
    session_mean_lb = None
    mean_gated_out = False
    needed_mean = None
    if session_failure_margin is not None and R > 1 and mean_quality_claim:
        session_mean_lb = (1.0 - q_upper) * lo + q_upper * SESSION_FLOOR
        mean_gated_out = session_mean_lb < -margin
        if mean_gated_out:
            # Sessions the mean claim needs if no further session fails, holding
            # the observed-session bound fixed (conservative: more sessions also
            # tighten lo). The mixed bound rises monotonically as sessions grow
            # (q_upper falls), so bisect rather than scan.
            def _mean_lb_at(rr: int) -> float:
                qu = binomial_upper(failures, rr, alpha)
                return (1.0 - qu) * lo + qu * SESSION_FLOOR

            cap = 20000
            if _mean_lb_at(cap) < -margin:
                needed_mean = None  # even a huge run cannot bound the loss
            else:
                a, b = R, cap
                while b - a > 1:
                    mid = (a + b) // 2
                    if _mean_lb_at(mid) >= -margin:
                        b = mid
                    else:
                        a = mid
                needed_mean = b
    common["session_mean_lb"] = session_mean_lb

    how = tail = ""
    if floored:
        how = f"{P} {unit}s show no variation" if se2 == 0.0 else f"{P} {unit}s are few"
        tail = (
            f"a difference within {bound:.3f} of the observed {grand:+.3f} stays "
            f"consistent with the data (rule of three at {P} {unit}s)"
        )
    if hi < -margin:
        verdict = BLOCK
        reason = (
            f"{how}, and {tail}, all of it worse than the -{margin:g} margin"
            if floored
            else f"the whole interval is worse than the -{margin:g} margin, so a "
            f"drop in {metric} that matters is demonstrated"
        )
    elif lo > -margin:
        verdict = PASS
        reason = (
            f"{how}, but {tail}, all of it inside the -{margin:g} margin"
            if floored
            else f"the whole interval is better than the -{margin:g} margin, so "
            f"non-inferiority of {metric} is demonstrated"
        )
        if mean_gated_out and not gated_out:
            # The frequency of failure sessions is bounded, but the MEAN is
            # not: an unobserved severe-session tail (up to q_upper of sessions,
            # each worth as much as -1) could pull the true mean past the
            # margin even though every observed session was mild. This is the
            # round-4 mixed-harm case. A demonstrated regression above still
            # blocks; a mild-looking sample does not earn a mean PASS.
            verdict = INCONCLUSIVE
            remedy = (
                f"about {needed_mean} sessions with no further failure would "
                f"bound the mean"
                if needed_mean is not None
                else "no session count bounds it while the observed loss stands"
            )
            reason = (
                f"non-inferior in the {R} session(s) sampled (observed mean "
                f"{grand:+.3f}), but an unobserved failure tail up to "
                f"{q_upper:.3f} of sessions leaves the mean quality delta as "
                f"low as {session_mean_lb:.3f}, which the -{margin:g} margin "
                f"does not cover; {remedy}"
            )
        elif gated_out:
            # The prompt-population interval clears, but the session-failure
            # rate is not bounded under the declared margin: too few sessions,
            # or failures were observed. A demonstrated regression above
            # still blocks; unobserved sessions cannot un-happen observed ones.
            verdict = INCONCLUSIVE
            remedy = (
                f"{needed} sessions with no further failure would bound it"
                if needed is not None
                else "the observed failure rate already exceeds the margin"
            )
            reason = (
                f"non-inferior on the prompts in the {R} session(s) observed, "
                f"but {failures} session failure(s) in {R} leave the "
                f"per-session failure rate bounded only at {q_upper:.3f}, above "
                f"the declared session-failure margin of "
                f"{session_failure_margin:g}; {remedy}"
            )
        elif diagnostic:
            verdict = INCONCLUSIVE
            reason = (
                f"observed-sessions diagnostic: would be non-inferior on the "
                f"prompts in the {R} session(s) observed, with {failures} "
                f"session failure(s) and a failure-rate bound of {q_upper:.3f}; "
                f"a diagnostic cannot approve a release"
            )
        elif session_failure_margin is not None:
            reason += (
                f"; {failures} session failure(s) in {R} bound the per-session "
                f"failure rate at {q_upper:.3f}, under the declared "
                f"session-failure margin of {session_failure_margin:g}"
            )
        else:
            reason += (
                f"; no session policy declared, so this covers only the "
                f"session conditions observed in {R} session(s): {failures} "
                f"session failure(s), rate bounded at {q_upper:.3f}. A release "
                f"gate must declare a session-failure margin"
            )
    else:
        verdict = INCONCLUSIVE
        if floored:
            reason = (
                f"{how}, and {tail}, which the -{margin:g} margin does not "
                f"cover; more {unit}s, not more repeats, would settle it"
            )
        else:
            what = (
                "session-to-session variation"
                if limited_by == "sessions"
                else "prompt-to-prompt variation"
            )
            more = "sessions" if limited_by == "sessions" else unit + "s"
            reason = (
                f"the interval straddles the -{margin:g} margin; {what} is what "
                f"limits it, so more {more} would narrow it"
            )
    return Decision(verdict, metric, P, grand, lo, hi, margin, reason, **common)


def decide_paired_binary(
    baseline: list[bool],
    candidate: list[bool],
    margin: float,
    metric: str = "success_rate",
    alpha: float = 0.05,
    session_failure_margin: float | None = None,
    diagnostic: bool = False,
    mean_quality_claim: bool = False,
) -> Decision:
    """Qualify a pass/fail metric measured once per item on the same items.

    The single-session form of :func:`decide_paired_grid`: each item is its
    own unit, there is no session component to estimate, and the decision
    says so. Used by command-axis adapters, where the items are parse
    boundaries or tool calls and the measurement runs once.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired inputs must describe the same items")
    pairs = {
        f"item{k:06d}": [(bool(b), bool(c))]
        for k, (b, c) in enumerate(zip(baseline, candidate, strict=True))
    }
    return decide_paired_grid(
        pairs,
        margin,
        metric=metric,
        alpha=alpha,
        unit="item",
        session_failure_margin=session_failure_margin,
        diagnostic=diagnostic,
        mean_quality_claim=mean_quality_claim,
    )


def gate(
    results: dict[str, tuple[list[float], float, bool]],
) -> tuple[str, list[Decision]]:
    """Combine per-metric decisions into one release verdict.

    A release is blocked if any metric blocks, and held if any metric is
    inconclusive. Only an all-PASS set ships, so an unmeasurable metric can
    never be silently treated as a passing one.
    """
    decisions = [
        decide(deltas, margin, metric=name, higher_is_better=hib)
        for name, (deltas, margin, hib) in results.items()
    ]
    return combine(d.verdict for d in decisions), decisions


def demo() -> None:
    """Self-check on synthetic cases with known answers."""
    # A clear regression, tight spread.
    d = decide([-5.0, -5.2, -4.8, -5.1, -4.9], margin=1.0, metric="good_rps")
    assert d.verdict == BLOCK, d

    # A clear non-regression, tight spread.
    d = decide([0.1, 0.0, -0.1, 0.05, -0.05], margin=1.0, metric="good_rps")
    assert d.verdict == PASS, d

    # Same mean as the PASS case but noisy: must not pass.
    d = decide([6.0, -6.0, 5.0, -5.0, 0.2], margin=1.0, metric="good_rps")
    assert d.verdict == INCONCLUSIVE, d
    assert d.runs_needed and d.runs_needed > 5, d

    # A result sitting almost exactly on the margin is expensive to resolve,
    # even though its interval is narrow.
    on_margin = decide(
        [-1.02, -0.98, -1.05, -0.95, -1.01], margin=1.0, metric="good_rps"
    )
    assert on_margin.verdict == INCONCLUSIVE, on_margin
    assert on_margin.runs_needed > 50, on_margin

    # A single run can never pass, however good it looks.
    d = decide([42.0], margin=1.0, metric="good_rps")
    assert d.verdict == INCONCLUSIVE, d

    # Latency: positive delta is worse, so this must block.
    d = decide(
        [9.0, 9.4, 9.1, 9.2, 9.3],
        margin=2.0,
        metric="p95_tpot_ms",
        higher_is_better=False,
    )
    assert d.verdict == BLOCK, d
    assert d.mean_delta > 0, d

    # One inconclusive metric holds the whole release.
    verdict, _ = gate(
        {
            "good_rps": ([0.1, 0.0, -0.1, 0.05, -0.05], 1.0, True),
            "p95_tpot_ms": ([6.0, -6.0, 5.0, -5.0, 0.2], 1.0, False),
        }
    )
    assert verdict == INCONCLUSIVE, verdict

    # Paired pass/fail metric: a clear reliability drop must block.
    before = [True] * 100
    after = [True] * 60 + [False] * 40
    d = decide_paired_binary(before, after, margin=0.05, metric="parse_survival")
    assert d.verdict == BLOCK, d
    assert d.mean_delta < -0.3, d

    # Identical outcomes are non-inferior.
    d = decide_paired_binary(before, list(before), margin=0.05)
    assert d.verdict == PASS, d

    # A drop close to the margin on few items must not pass.
    d = decide_paired_binary([True] * 20, [True] * 18 + [False] * 2, margin=0.05)
    assert d.verdict == INCONCLUSIVE, d

    # Degenerate samples: agreeing arms only demonstrate non-inferiority once
    # there are enough pairs for the rule-of-three bound to fit the margin.
    d = decide_paired_binary([True] * 2, [True] * 2, margin=0.01)
    assert d.verdict == INCONCLUSIVE, d
    assert d.runs_needed is None, d
    d = decide_paired_binary([True] * 1000, [True] * 1000, margin=0.01)
    assert d.verdict == PASS, d
    # Agreeing at a ceiling is fine; agreeing at a floor is still the
    # broken-grader case and must stay inconclusive.
    d = decide_paired_binary([False] * 1000, [False] * 1000, margin=0.01)
    assert d.verdict == INCONCLUSIVE, d

    # A metric that never succeeded in either arm measured nothing, and must
    # not be reported as non-inferior.
    d = decide_paired_binary(
        [False] * 40, [False] * 40, margin=0.05, metric="answer_quality"
    )
    assert d.verdict == INCONCLUSIVE, d
    assert "never succeeded" in d.reason, d

    # Inputs a caller can get wrong must fail loudly, never fall through to a
    # verdict.
    for bad in (
        lambda: decide([1.0, 2.0], margin=0.0),
        lambda: decide_paired_binary([True], [True, False], margin=0.1),
        lambda: decide_paired_binary([True] * 3, [True] * 3, margin=0.1, alpha=0.0),
        lambda: decide_paired_binary([True] * 3, [True] * 3, margin=0.1, alpha=1.0),
        lambda: decide_paired_binary([True] * 3, [True] * 3, margin=-0.1),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input reached a verdict")

    # No data is not evidence of anything.
    d = decide([], margin=1.0)
    assert d.verdict == INCONCLUSIVE and d.n_runs == 0, d
    d = decide_paired_binary([], [], margin=0.1)
    assert d.verdict == INCONCLUSIVE, d

    # A latency-like improvement passes and is reported in the caller's sign.
    d = decide(
        [-3.0, -3.2, -2.9, -3.1, -3.0],
        margin=1.0,
        metric="p95_ttft_ms",
        higher_is_better=False,
    )
    assert d.verdict == PASS and d.mean_delta < 0, d

    # Release-level precedence: BLOCK beats INCONCLUSIVE beats PASS.
    verdict, _ = gate(
        {
            "a": ([0.1, 0.0, -0.1, 0.05, -0.05], 1.0, True),  # PASS
            "b": ([6.0, -6.0, 5.0, -5.0, 0.2], 1.0, True),  # INCONCLUSIVE
            "c": ([-5.0, -5.2, -4.8, -5.1, -4.9], 1.0, True),  # BLOCK
        }
    )
    assert verdict == BLOCK, verdict

    # A mean sitting exactly on the boundary cannot be settled by more runs.
    assert runs_needed(sd=0.5, margin=1.0, mean_delta=-1.0) == 500

    # A tighter alpha widens the interval, so a borderline block becomes
    # inconclusive rather than flipping to a pass.
    borderline = ([True] * 100, [True] * 84 + [False] * 16)
    wide = decide_paired_binary(*borderline, margin=0.1, alpha=0.05)
    tight = decide_paired_binary(*borderline, margin=0.1, alpha=0.001)
    assert tight.ci_lo <= wide.ci_lo and tight.ci_hi >= wide.ci_hi, (wide, tight)

    # The t quantile replaces the alpha=0.05 table and must reproduce it,
    # and it must be right at the Bonferroni-sized tails the families use.
    for df, want in ((1, 12.706), (4, 2.776), (10, 2.228), (30, 2.042)):
        assert abs(t_quantile(0.975, df) - want) < 2e-3, (df, t_quantile(0.975, df))
    assert abs(t_quantile(0.975, 100000) - 1.960) < 2e-3
    for q, df, want in (
        (0.995, 4, 4.604),
        (0.995, 10, 3.169),
        (0.999, 5, 5.893),
        (0.9995, 30, 3.646),
    ):
        assert abs(t_quantile(q, df) - want) < 3e-3, (q, df, t_quantile(q, df))

    # Prompt-population claims. Copies of a prompt in further sessions are
    # not further prompts: the interval over 20 prompts is identical whether
    # each was run once or five times, and only the observation count moves.
    once = {f"p{i}": [(True, i % 4 != 0)] for i in range(20)}
    five = {k: v * 5 for k, v in once.items()}
    a = decide_paired_grid(once, margin=0.10, metric="q")
    b = decide_paired_grid(five, margin=0.10, metric="q")
    assert (a.ci_lo, a.ci_hi, a.verdict) == (b.ci_lo, b.ci_hi, b.verdict), (a, b)
    assert a.unique_prompts == b.unique_prompts == 20, (a, b)
    assert (a.observations, b.observations) == (20, 100), (a, b)
    assert (a.repeats, b.repeats) == (1, 5), (a, b)

    # Zero disagreement: the rule-of-three bound counts prompts. Twenty
    # agreeing prompts run five times are 100 observations that settle
    # nothing at a 0.10 margin (bound 3/20); forty prompts do (3/40).
    agree20 = {f"p{i}": [(True, True)] * 5 for i in range(20)}
    d = decide_paired_grid(agree20, margin=0.10)
    assert d.verdict == INCONCLUSIVE and d.observations == 100, d
    assert d.runs_needed is None, d
    agree40 = {f"p{i}": [(True, True)] * 5 for i in range(40)}
    assert decide_paired_grid(agree40, margin=0.10).verdict == PASS

    # A shock that hits one whole session: every prompt shows the same -0.2,
    # so prompt-only variance is zero and a prompt-only interval would
    # BLOCK. Five sessions cannot tell that from the change; the session
    # component has to widen it to INCONCLUSIVE and say what limited it.
    shock = {
        f"p{i}": [(True, True), (True, True), (True, False), (True, True), (True, True)]
        for i in range(100)
    }
    d = decide_paired_grid(shock, margin=0.10)
    assert d.verdict == INCONCLUSIVE, d
    assert d.limited_by == "sessions", d
    assert d.components["session"] > d.components["prompt"], d

    # A prompt-level regression the sessions agree on blocks: 16 of 40
    # prompts wrong in every session.
    consistent = {f"p{i}": [(True, i % 5 >= 2)] * 5 for i in range(40)}
    d = decide_paired_grid(consistent, margin=0.10)
    assert d.verdict == BLOCK and d.limited_by == "prompts", d
    assert (d.unique_prompts, d.repeats, d.observations) == (40, 5, 200), d
    assert d.unit == "prompt" and d.alpha == 0.05 and d.seed is None, d

    # A ragged grid is a pairing fault, never data.
    try:
        decide_paired_grid(
            {"a": [(True, True)] * 5, "b": [(True, True)] * 4}, margin=0.1
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a ragged grid reached a verdict")

    # Single-session data cannot estimate a session component and says so.
    d = decide_paired_binary([True] * 100, [True] * 90 + [False] * 10, margin=0.05)
    assert d.repeats == 1 and d.components["session"] is None, d
    assert d.unit == "item", d

    # Release-level contract, one function for every layer that combines
    # verdicts: a measurement ERROR outranks everything, then any BLOCK blocks,
    # then any INCONCLUSIVE holds, and only an all-PASS set passes. An empty
    # set measured nothing and cannot pass.
    for verdicts, expected in (
        ([PASS, PASS], PASS),
        ([PASS, INCONCLUSIVE], INCONCLUSIVE),
        ([INCONCLUSIVE, PASS, PASS], INCONCLUSIVE),
        ([PASS, BLOCK], BLOCK),
        ([INCONCLUSIVE, BLOCK], BLOCK),
        ([BLOCK, INCONCLUSIVE, PASS], BLOCK),
        ([PASS, ERROR], ERROR),
        ([BLOCK, ERROR], ERROR),
        ([INCONCLUSIVE, ERROR, BLOCK], ERROR),
        ([], INCONCLUSIVE),
    ):
        assert combine(verdicts) == expected, (verdicts, expected)
    try:
        combine([PASS, "SHIPPED"])
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown verdict reached a release decision")

    # A non-finite measurement is a measurement that did not happen. It is
    # ERROR, never a verdict, and in particular never PASS or INCONCLUSIVE.
    for bad in ([1.0, float("nan"), 1.2], [float("inf"), 0.1, 0.2]):
        d = decide(bad, margin=1.0, metric="p95_ttft_ms", higher_is_better=False)
        assert d.verdict == ERROR, d
        assert "not finite" in d.reason, d
    assert combine([ERROR, PASS]) == ERROR

    # Session risk (review finding P1-3, 2026-09-05). Five sessions that all
    # agree cannot exclude an intermittent whole-session effect: with zero
    # occurrences in R sessions the one-sided 95% upper bound on the
    # per-session rate is 1 - alpha^(1/R), 0.451 at R=5. Every decision
    # reports that bound. By default the claim is scoped to the sessions
    # observed and the bound is disclosed; with a declared session-failure
    # margin the decision cannot PASS until R sessions bound the rate under
    # it, and it says how many sessions that takes.
    agree = {f"p{i}": [(True, True)] * 5 for i in range(800)}
    d = decide_paired_grid(agree, margin=0.10)
    assert d.verdict == PASS, d
    assert abs(d.session_rate_upper - (1 - 0.05**0.2)) < 1e-12, d
    assert d.sessions_needed is None, d
    d = decide_paired_grid(agree, margin=0.10, session_failure_margin=0.10)
    assert d.verdict == INCONCLUSIVE, d
    assert d.sessions_needed == 29, d
    assert "session failure" in d.reason and "29 sessions" in d.reason, d.reason
    agree29 = {f"p{i}": [(True, True)] * 29 for i in range(800)}
    d = decide_paired_grid(agree29, margin=0.10, session_failure_margin=0.10)
    assert d.verdict == PASS, d
    assert d.session_rate_upper <= 0.10, d
    # A regression demonstrated in the observed sessions still blocks.
    worse = {f"p{i}": [(True, i % 5 >= 2)] * 5 for i in range(40)}
    d = decide_paired_grid(worse, margin=0.10, session_failure_margin=0.10)
    assert d.verdict == BLOCK, d
    # One session excludes nothing.
    d = decide_paired_binary([True] * 100, [True] * 100, margin=0.05)
    assert abs(d.session_rate_upper - 0.95) < 1e-12, d
    d = decide_paired_binary(
        [True] * 100, [True] * 100, margin=0.05, session_failure_margin=0.5
    )
    assert d.verdict == INCONCLUSIVE and d.sessions_needed == 5, d

    # Round-2 finding 1: the bound must count observed failures. One failed
    # session in 31 leaves an exact one-sided 95% upper bound of 0.144 on the
    # failure rate, which does not exclude 0.10; the zero-occurrence formula
    # said 0.092 and the gate passed. A session is a failure event when its
    # own mean delta is past the margin.
    assert abs(binomial_upper(0, 5, 0.05) - (1 - 0.05**0.2)) < 1e-9
    assert abs(binomial_upper(1, 31, 0.05) - 0.14409) < 5e-5, binomial_upper(
        1, 31, 0.05
    )
    assert binomial_upper(0, 29, 0.05) <= 0.10 < binomial_upper(0, 28, 0.05)
    one_failed = {f"p{i}": [(True, True)] * 30 + [(True, False)] for i in range(800)}
    d = decide_paired_grid(one_failed, margin=0.10, session_failure_margin=0.10)
    assert d.verdict == INCONCLUSIVE, d
    assert d.session_failures_observed == 1, d
    assert abs(d.session_rate_upper - 0.14409) < 5e-5, d
    assert d.sessions_needed == 46, d
    one_in_46 = {f"p{i}": [(True, True)] * 45 + [(True, False)] for i in range(800)}
    d = decide_paired_grid(one_in_46, margin=0.10, session_failure_margin=0.10)
    assert d.verdict == PASS and d.session_rate_upper <= 0.10, d
    half = {f"p{i}": [(True, True)] * 30 + [(True, i % 2 == 0)] for i in range(800)}
    d = decide_paired_grid(half, margin=0.10, session_failure_margin=0.10)
    assert d.session_failures_observed == 1, d
    mild = {f"p{i}": [(True, True)] * 30 + [(True, i % 20 != 0)] for i in range(800)}
    d = decide_paired_grid(mild, margin=0.10, session_failure_margin=0.10)
    assert d.session_failures_observed == 0 and d.verdict == PASS, d
    # Five failed sessions in ten: the interval blocks on its own, and the
    # count says a failure rate above one half stays consistent with the data.
    many = {f"p{i}": [(True, True)] * 5 + [(True, False)] * 5 for i in range(100)}
    d = decide_paired_grid(many, margin=0.10, session_failure_margin=0.10)
    assert d.verdict == BLOCK and d.session_failures_observed == 5, d
    assert d.session_rate_upper > 0.5, d

    # Round-4 mean-quality tail (2026-09-09). A candidate persistently 7 points
    # worse whose 4% whole-session outage did not land in the sample: the
    # observed sessions all read -0.07, the observed failure count is 0, but the
    # true mean is -0.107. The mean claim must charge the unobserved failure
    # tail: (1 - q_upper)*lo + q_upper*(-1). Without mean_quality_claim the old
    # behaviour stands (frequency guard only); with it, the mean is gated.
    mild = {f"p{i}": [(True, i >= 56)] * 29 for i in range(800)}
    d = decide_paired_grid(mild, margin=0.10, session_failure_margin=0.10)
    assert d.verdict == PASS, d  # frequency-only claim still passes
    assert d.session_mean_lb is None, d
    d = decide_paired_grid(
        mild, margin=0.10, session_failure_margin=0.10, mean_quality_claim=True
    )
    assert d.verdict == INCONCLUSIVE, d
    assert d.session_failures_observed == 0, d
    assert d.session_mean_lb is not None and d.session_mean_lb < -0.10, d
    assert (
        abs(
            d.session_mean_lb
            - ((1 - d.session_rate_upper) * d.ci_lo + d.session_rate_upper * -1.0)
        )
        < 1e-9
    ), d
    assert "unobserved failure tail" in d.reason, d.reason
    # A clean change: the mean tail needs enough sessions. 29 is not enough
    # (the frequency minimum does not buy the mean); for this 800-prompt
    # fixture at alpha 0.05 and a 0.10 policy the boundary is 30. It is not a
    # general number: it moves with the policy, alpha and the prompt floor.
    for R, want in ((29, INCONCLUSIVE), (30, PASS), (45, PASS)):
        clean = {f"p{i}": [(True, True)] * R for i in range(800)}
        d = decide_paired_grid(
            clean, margin=0.10, session_failure_margin=0.10, mean_quality_claim=True
        )
        assert d.verdict == want, (R, d)
    # At the twelve-family alpha a 17-prompt family with no observed difference
    # never clears a 0.10 margin, however many sessions: the prompt floor
    # (-ln(0.05/12)/17 = 0.32) pins the interval's lower edge at -0.32 and the
    # tail term only lowers it. Sessions cannot buy prompts.
    family_policy = dict(
        margin=0.10,
        alpha=0.05 / 12,
        session_failure_margin=0.10,
        mean_quality_claim=True,
    )
    small = {f"p{i}": [(True, True)] * 80 for i in range(17)}
    d = decide_paired_grid(small, **family_policy)
    assert d.verdict == INCONCLUSIVE, d
    # That is a limit on a family showing no difference, not on 17 prompts: a
    # large observed improvement (4 of 17 correct becoming 17 of 17) can pass,
    # but only once the sessions bound the failure rate under the policy at
    # that alpha, which first happens at 53 sessions.
    for R, want in ((29, INCONCLUSIVE), (52, INCONCLUSIVE), (53, PASS), (80, PASS)):
        improved = {f"p{i}": [(i < 4, True)] * R for i in range(17)}
        d = decide_paired_grid(improved, **family_policy)
        assert d.verdict == want, (R, d)
    # An observed outage session blocks or holds regardless: a demonstrated
    # regression is unaffected by the tail term.
    outage = {f"p{i}": [(True, i >= 56)] * 28 + [(True, False)] for i in range(800)}
    d = decide_paired_grid(
        outage, margin=0.10, session_failure_margin=0.10, mean_quality_claim=True
    )
    assert d.verdict in (BLOCK, INCONCLUSIVE) and d.session_failures_observed == 1, d
    # The mean tail applies only where asked: success/compliance keep the
    # frequency-only claim, so a clean 29-session reliability check still passes.
    d = decide_paired_grid(
        {f"p{i}": [(True, True)] * 29 for i in range(800)},
        margin=0.01,
        session_failure_margin=0.10,
    )
    assert d.verdict == PASS and d.session_mean_lb is None, d

    print("release_gate self-check: all cases pass")


if __name__ == "__main__":
    demo()
