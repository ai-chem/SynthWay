from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent
DEFAULT_RUN_DIRS = [
    RUNS_DIR / "repeat_1",
    RUNS_DIR / "repeat_2",
    RUNS_DIR / "repeat_3",
]
DEFAULT_OUT_DIR = (
    RUNS_DIR / "cross_judge_expert_faithfulness_three_run_stats"
)

METRICS = ["chemical_geval", "answer_relevancy", "faithfulness"]
METRIC_LABELS = {
    "chemical_geval": "Chemical plausibility",
    "answer_relevancy": "Answer relevancy",
    "faithfulness": "New Faithfulness",
}
ROUTE_KEY = ["arm_key", "case_id", "route_name"]
N_BOOT = 20000
SEED = 20260904


# loading


def parse_bool(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    values = series.astype(str).str.strip().str.lower().map(mapping)
    if values.isna().any():
        bad = sorted(series.loc[values.isna()].astype(str).unique())
        raise ValueError(f"Cannot parse boolean values: {bad}")
    return values.astype(bool)


def load_runs(run_dirs: Sequence[Path], labels: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts, routes = [], []
    for run_dir, label in zip(run_dirs, labels):
        run_dir = run_dir.expanduser().resolve()
        a = pd.read_csv(run_dir / "tables" / "attempt_trajectories_long.csv")
        r = pd.read_csv(run_dir / "tables" / "route_effects.csv")
        a["run"], r["run"] = label, label
        a["attempt_no"] = a["attempt_no"].astype(int)
        a["score"] = pd.to_numeric(a["score"], errors="coerce")
        a["passed_all"] = parse_bool(a["passed_all"])
        r["initial_pass_all"] = parse_bool(r["initial_pass_all"])
        r["final_pass_all"] = parse_bool(r["final_pass_all"])
        attempts.append(a)
        routes.append(r)
    return pd.concat(attempts, ignore_index=True), pd.concat(routes, ignore_index=True)


# inference


def cluster_bootstrap(
    values: pd.Series,
    clusters: pd.Series,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, float]:
    """Percentile bootstrap for a mean, resampling whole clusters.

    ``values`` are route-level statistics (already averaged over repeats) and
    ``clusters`` is the cluster label of each value.  Whole clusters are drawn
    with replacement, so both between-cluster and within-cluster correlation
    are preserved.  Percentile intervals are used because the estimand is a
    proportion that can sit close to 0 or 1, where a Wald interval leaves the
    unit interval.
    """
    frame = pd.DataFrame({"value": values.to_numpy(float), "cluster": clusters.to_numpy()})
    frame = frame.dropna()
    groups = [g["value"].to_numpy() for _, g in frame.groupby("cluster", sort=True)]
    n_clusters = len(groups)
    if n_clusters == 0:
        return {"mean": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan, "n_clusters": 0}

    rng = np.random.default_rng(seed)
    point = frame["value"].mean()
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_clusters, n_clusters)
        draws[b] = np.concatenate([groups[i] for i in idx]).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(point),
        "se": float(draws.std(ddof=1)),
        "lo": float(lo),
        "hi": float(hi),
        "n_clusters": n_clusters,
        "n_units": int(len(frame)),
    }


def route_level(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Average repeats within a route (Miller rec. 3) and keep the cluster id."""
    return (
        frame.groupby(ROUTE_KEY + ["generator_key", "judge_key"], as_index=False)
        .agg(value=(value, "mean"), n_runs=("run", "nunique"))
        .rename(columns={"value": value})
    )


def summarize(frame: pd.DataFrame, value: str, label: str, cluster: str = "case_id") -> dict:
    out = cluster_bootstrap(frame[value], frame[cluster])
    out["subset"] = label
    return out


# ---------------------------------------------------------------- null model


def beta_binomial_null(k: np.ndarray, n: np.ndarray, n_draws: int) -> dict[str, float]:
    """Expected pass rate from ``n_draws`` independent judge evaluations.

    ``k`` successes out of ``n`` independent evaluations of the *unmodified*
    route give a per-route pass probability.  Point estimates from three draws
    are far too noisy to plug into 1-(1-p)^4 directly (the map is concave, so
    the plug-in estimate is biased low), so an empirical-Bayes Beta prior is
    fitted across routes and the posterior predictive is used instead.
    """

    def nll(par: np.ndarray) -> float:
        a, b = np.exp(par)
        return float(-np.sum(special.betaln(a + k, b + n - k) - special.betaln(a, b)))

    fit = optimize.minimize(nll, np.zeros(2), method="Nelder-Mead")
    a, b = np.exp(fit.x)
    post_a, post_b = a + k, b + n - k
    # E[(1-p)^n_draws] under Beta(post_a, post_b)
    q = np.exp(special.betaln(post_a, post_b + n_draws) - special.betaln(post_a, post_b))
    return {
        "alpha": float(a),
        "beta": float(b),
        "prior_mean": float(a / (a + b)),
        "single_eval_pass": float(k.sum() / n.sum()),
        "null_pass_at_k": float(np.mean(1.0 - q)),
        "n_draws": int(n_draws),
    }


# analyses


def analyse_pass(attempts: pd.DataFrame, routes: pd.DataFrame, max_regens: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative pass by attempt, with cluster-bootstrap intervals."""
    attempt_id = ["run"] + ROUTE_KEY + ["attempt_no"]
    status = attempts.groupby(attempt_id, as_index=False).agg(passed=("passed_all", "first"))
    passed_at = {
        key: set(g.loc[g["passed"], "attempt_no"].astype(int))
        for key, g in status.groupby(["run"] + ROUTE_KEY, sort=False)
    }

    rows = []
    meta = routes[["run"] + ROUTE_KEY + ["generator_key", "judge_key"]]
    for row in meta.itertuples(index=False):
        key = (row.run, row.arm_key, row.case_id, row.route_name)
        hits = passed_at.get(key, set())
        for step in range(max_regens + 1):
            rows.append(
                {
                    "run": row.run,
                    "arm_key": row.arm_key,
                    "case_id": row.case_id,
                    "route_name": row.route_name,
                    "generator_key": row.generator_key,
                    "judge_key": row.judge_key,
                    "step": step,
                    "passed": bool(any(n <= step + 1 for n in hits)),
                }
            )
    traj = pd.DataFrame(rows)

    results = []
    for step, g in traj.groupby("step"):
        rl = route_level(g, "passed")
        for cluster, name in (("case_id", "case"), ("route_name_unit", "route")):
            if name == "route":
                rl = rl.assign(route_name_unit=rl[ROUTE_KEY].agg("|".join, axis=1))
            res = summarize(rl, "passed", f"step={step}", cluster=cluster)
            res.update({"step": step, "cluster_level": name, "scope": "overall"})
            results.append(res)
        for role, col in (("generator", "generator_key"), ("judge", "judge_key")):
            for model, sub in rl.groupby(col):
                res = summarize(sub, "passed", f"step={step}", cluster="case_id")
                res.update({"step": step, "cluster_level": "case", "scope": f"{role}={model}"})
                results.append(res)
    return traj, pd.DataFrame(results)


def analyse_paired(traj: pd.DataFrame, max_regens: int) -> pd.DataFrame:
    """Paired initial-vs-final pass change, at route and at case level."""
    first = traj[traj.step == 0].set_index(["run"] + ROUTE_KEY)["passed"].astype(int)
    last = traj[traj.step == max_regens].set_index(["run"] + ROUTE_KEY)["passed"].astype(int)
    gain = (last - first).rename("gain").reset_index()

    out = []
    # route level: average the three repeats within route, then a sign test
    rl = gain.groupby(ROUTE_KEY, as_index=False).agg(gain=("gain", "mean"))
    rl["case_id"] = rl["case_id"]
    pos, neg = int((rl.gain > 0).sum()), int((rl.gain < 0).sum())
    p_exact = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan
    boot = cluster_bootstrap(rl["gain"], rl["case_id"])
    out.append(
        {
            "level": "route (n=%d), clustered on case" % len(rl),
            "n_units": len(rl),
            "improved": pos,
            "worsened": neg,
            "unchanged": int(len(rl) - pos - neg),
            "mean_gain": boot["mean"],
            "ci_lo": boot["lo"],
            "ci_hi": boot["hi"],
            "p_exact_sign": float(p_exact),
        }
    )
    # case level: fully independent units
    cl = gain.groupby("case_id", as_index=False).agg(gain=("gain", "mean"))
    pos, neg = int((cl.gain > 0).sum()), int((cl.gain < 0).sum())
    p_exact = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else np.nan
    w = stats.wilcoxon(cl.gain[cl.gain != 0]) if (cl.gain != 0).sum() > 0 else None
    out.append(
        {
            "level": "case (n=%d), independent" % len(cl),
            "n_units": len(cl),
            "improved": pos,
            "worsened": neg,
            "unchanged": int(len(cl) - pos - neg),
            "mean_gain": float(cl.gain.mean()),
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "p_exact_sign": float(p_exact),
            "p_wilcoxon": float(w.pvalue) if w is not None else np.nan,
        }
    )
    return pd.DataFrame(out)


def analyse_selection_bias(attempts: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Compare the selected 'best' delta against deltas that are not selected.

    Three estimands, all relative to attempt 1:
      selected   : best-of-k, argmax over the reported judge scores.
      one_regen  : attempt 2, restricted to routes whose attempt 1 failed.  The
                   restriction defines the population the pipeline acts on and
                   attempt 2 is never chosen on its outcome, so this is clean.
      exhausted  : attempt 4, restricted to routes that used all four attempts
                   (i.e. never passed).  No stopping-rule selection applies.
    """
    wide = attempts.pivot_table(
        index=["run"] + ROUTE_KEY + ["metric"], columns="attempt_no", values="score"
    )
    status = (
        attempts.groupby(["run"] + ROUTE_KEY + ["attempt_no"], as_index=False)
        .agg(passed=("passed_all", "first"))
    )
    init_fail = (
        status[status.attempt_no == 1].set_index(["run"] + ROUTE_KEY)["passed"].pipe(lambda s: ~s)
    )
    used = routes.set_index(["run"] + ROUTE_KEY)["attempts_used"]

    rows = []
    for metric in METRICS:
        sub = wide.xs(metric, level="metric")
        sel = routes.set_index(["run"] + ROUTE_KEY)
        delta_sel = (sel[f"final_{metric}"] - sel[f"initial_{metric}"]).rename("d").reset_index()
        delta_sel["case_id"] = delta_sel["case_id"]
        rl = delta_sel.groupby(ROUTE_KEY, as_index=False).agg(d=("d", "mean"))
        b = cluster_bootstrap(rl["d"], rl["case_id"])
        rows.append({"metric": metric, "estimand": "selected best-of-k", **b})

        mask = init_fail.reindex(sub.index).fillna(False)
        one = (sub[2] - sub[1])[mask & sub[2].notna()].rename("d").reset_index()
        rl = one.groupby(ROUTE_KEY, as_index=False).agg(d=("d", "mean"))
        b = cluster_bootstrap(rl["d"], rl["case_id"])
        rows.append({"metric": metric, "estimand": "one regeneration (unselected)", **b})

        mask4 = used.reindex(sub.index) == 4
        ex = (sub[4] - sub[1])[mask4.fillna(False) & sub[4].notna()].rename("d").reset_index()
        rl = ex.groupby(ROUTE_KEY, as_index=False).agg(d=("d", "mean"))
        b = cluster_bootstrap(rl["d"], rl["case_id"])
        rows.append({"metric": metric, "estimand": "attempt 4, exhausted (unselected)", **b})
    return pd.DataFrame(rows)


def analyse_judge_reliability(attempts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Test-retest of the judge on identical text (attempt 1 is never modified)."""
    a1 = attempts[attempts.attempt_no == 1]
    wide = a1.pivot_table(index=ROUTE_KEY + ["metric"], columns="run", values="score")
    rows = []
    for metric in METRICS:
        s = wide.xs(metric, level="metric").dropna()
        sd = s.std(axis=1, ddof=1)
        rows.append(
            {
                "metric": metric,
                "n_routes": len(s),
                "mean_score": float(s.to_numpy().mean()),
                "mean_within_route_sd": float(sd.mean()),
                "max_within_route_sd": float(sd.max()),
                "pct_identical_all_runs": float(100 * (sd < 1e-9).mean()),
            }
        )
    reliability = pd.DataFrame(rows)

    passed = (
        a1.drop_duplicates(["run"] + ROUTE_KEY)
        .pivot_table(index=ROUTE_KEY, columns="run", values="passed_all")
    )
    k = passed.sum(axis=1).to_numpy(float)
    n = np.full_like(k, passed.shape[1])
    null = beta_binomial_null(k, n, n_draws=4)
    return reliability, null


# report


def write_report(path: Path, blocks: dict) -> None:
    pass_tab = blocks["pass"]
    overall = pass_tab[(pass_tab.scope == "overall") & (pass_tab.cluster_level == "case")]
    null = blocks["null"]
    lines = [
        "# Cross-judge regeneration: inferential analysis",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Unit of analysis: synthesis route. Repeats are averaged within route "
        "(Miller 2024, rec. 3). Uncertainty is a percentile cluster bootstrap "
        f"over source systems (n={int(overall.n_clusters.iloc[0])} cases), "
        f"{N_BOOT} resamples.",
        "",
        "## 1. Cumulative pass rate by attempt",
        "",
        "| step | pass rate | 95% CI | clustered SE |",
        "| --- | --- | --- | --- |",
    ]
    for _, r in overall.sort_values("step").iterrows():
        lines.append(
            f"| {int(r.step)} | {100*r['mean']:.1f}% | "
            f"[{100*r.lo:.1f}, {100*r.hi:.1f}] | {100*r.se:.1f} pp |"
        )
    lines += [
        "",
        "## 2. Judge-noise null model",
        "",
        "Attempt 1 is the same unmodified route text in all three repeats, so the "
        "repeats measure judge test-retest noise directly.",
        "",
        "| quantity | value |",
        "| --- | --- |",
        f"| single-evaluation pass rate | {100*null['single_eval_pass']:.1f}% |",
        f"| expected pass from 4 independent evaluations of the *unchanged* route | {100*null['null_pass_at_k']:.1f}% |",
        f"| observed pass after 3 regenerations | {100*overall['mean'].iloc[-1]:.1f}% |",
        f"| naive gain over initial | {100*(overall['mean'].iloc[-1]-null['single_eval_pass']):+.1f} pp |",
        f"| gain over the judge-noise null | {100*(overall['mean'].iloc[-1]-null['null_pass_at_k']):+.1f} pp |",
        "",
        "## 3. Judge test-retest reliability (identical input, temperature 0)",
        "",
        blocks["reliability"].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 4. Paired initial-vs-final comparison",
        "",
        blocks["paired"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. Selection-bias control on metric deltas",
        "",
        "`selected best-of-k` maximises over the same judge scores that are then "
        "reported; the other two rows are not selected on the outcome. The gap "
        "between them is the magnitude of the winner's curse.",
        "",
        blocks["selection"][
            ["metric", "estimand", "mean", "se", "lo", "hi", "n_units"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path)
    parser.add_argument("--run-label", action="append")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-regens", type=int, default=3)
    args = parser.parse_args()

    run_dirs = args.run_dir or DEFAULT_RUN_DIRS
    labels = args.run_label or [f"v{i}" for i in range(1, len(run_dirs) + 1)]
    out_dir = args.out_dir.expanduser().resolve()
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    attempts, routes = load_runs(run_dirs, labels)
    traj, pass_tab = analyse_pass(attempts, routes, args.max_regens)
    paired = analyse_paired(traj, args.max_regens)
    selection = analyse_selection_bias(attempts, routes)
    reliability, null = analyse_judge_reliability(attempts)

    pass_tab.to_csv(out_dir / "tables" / "pass_rate_cluster_bootstrap.csv", index=False)
    paired.to_csv(out_dir / "tables" / "paired_initial_vs_final.csv", index=False)
    selection.to_csv(out_dir / "tables" / "selection_bias_control.csv", index=False)
    reliability.to_csv(out_dir / "tables" / "judge_test_retest.csv", index=False)
    (out_dir / "null_model.json").write_text(json.dumps(null, indent=2), encoding="utf-8")

    write_report(
        out_dir / "report.md",
        {
            "pass": pass_tab,
            "paired": paired,
            "selection": selection,
            "reliability": reliability,
            "null": null,
        },
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
