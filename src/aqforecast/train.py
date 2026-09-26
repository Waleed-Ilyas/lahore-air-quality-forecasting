"""End-to-end training pipeline with MLflow tracking.

Usage:  python -m aqforecast.train

Steps
1. load + validate + clean the data
2. build (origin, horizon) dataset
3. random-search hyper-parameters with blocked, expanding-window time-series CV (median model)
4. conformalised quantile regression (CQR): out-of-fold residuals from the CV widen the 80% band
5. fit on everything before the final hold-out year and evaluate on the hold-out
   against persistence, seasonal-naive and Holt-Winters baselines
6. refit on ALL data for the deployed model and save small artifacts
"""
from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import features as F
from . import ingest, quality
from . import models as M

ORIGIN_STEP = 3  # use every 3rd hour as a forecast origin (neighbouring origins are near-duplicates)
CV_FOLDS, CV_BLOCK_DAYS = 4, 120
N_TRIALS = 8
HW_ORIGINS = 150  # Holt-Winters is slow; evaluate it (and everything else) on a shared sample


def search_space(rng: np.random.Generator) -> dict:
    choice = [(300, 0.05), (500, 0.05), (600, 0.03)][rng.integers(3)]
    return {
        "n_estimators": choice[0],
        "learning_rate": choice[1],
        "num_leaves": int(rng.choice([15, 31, 63])),
        "min_child_samples": int(rng.choice([50, 100, 200, 400])),
        "feature_fraction": float(rng.choice([0.5, 0.7, 0.9])),
        "lambda_l2": float(rng.choice([0.0, 1.0, 10.0])),
    }


def cv_folds(meta: pd.DataFrame, test_start: pd.Timestamp):
    """Blocked expanding-window folds. Train rows must have their whole target before the block."""
    for j in range(CV_FOLDS):
        v_start = test_start - pd.Timedelta(days=CV_BLOCK_DAYS * (CV_FOLDS - j))
        v_end = v_start + pd.Timedelta(days=CV_BLOCK_DAYS)
        train = (meta["target_time"] < v_start).to_numpy()
        val = ((meta["origin"] >= v_start) & (meta["origin"] < v_end)).to_numpy()
        yield j, train, val


def cv_score(X, meta, folds, params, quantiles=(0.5,)):
    """Mean MAE (original units) of the median forecast across folds; also returns OOF preds."""
    maes, oof = [], []
    for _j, tr, va in folds:
        f = M.Forecaster(quantiles=quantiles, params=params).fit(X[tr], meta.y[tr])
        pred = f.predict(X[va])
        maes.append(float(np.abs(pred.iloc[:, len(quantiles) // 2].to_numpy() - meta.y[va]).mean()))
        oof.append(pd.concat([meta[va].reset_index(drop=True), pred], axis=1))
    return maes, pd.concat(oof, ignore_index=True)


def cqr_offsets(oof: pd.DataFrame) -> dict[str, float]:
    """Per-horizon-bucket CQR correction so the [q10, q90] band reaches ~80% coverage."""
    out = {}
    bucket = M._bucket_of(oof["horizon"].to_numpy())
    score = np.maximum(oof["q10"] - oof["y"], oof["y"] - oof["q90"]).to_numpy()
    for b in range(len(M.BUCKETS)):
        s = score[bucket == b]
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * 0.8) / n)
        out[str(b)] = float(np.quantile(s, level))
    return out


def main() -> None:
    t_start = time.time()
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{C.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("lahore-pm25-forecast")

    raw = ingest.load_history()
    report = quality.check(raw)
    df = quality.clean(raw)
    df = df.loc[df["pm2_5"].first_valid_index() :]  # first 77h of 2022-08-01 have no pollutants
    (C.ARTIFACTS_DIR / "quality_report.json").write_text(json.dumps(report, indent=1))

    test_start = df.index.max() - pd.Timedelta(days=C.TEST_DAYS)
    X, meta = F.build_dataset(df, origin_step=ORIGIN_STEP)
    print(f"dataset {X.shape}, test starts {test_start}")

    with mlflow.start_run(run_name="pipeline") as parent:
        mlflow.log_params({"origin_step": ORIGIN_STEP, "test_days": C.TEST_DAYS,
                           "cv_folds": CV_FOLDS, "cv_block_days": CV_BLOCK_DAYS,
                           "n_features": X.shape[1], "rows": len(X)})
        mlflow.log_dict(report, "quality_report.json")

        # ---- 1. hyper-parameter search on blocked time-series CV (before the hold-out year) ----
        folds = list(cv_folds(meta, test_start))
        sub = (np.arange(len(meta)) % 2 == 0)  # speed: search on half the origins
        Xs, ms = X[sub].reset_index(drop=True), meta[sub].reset_index(drop=True)
        folds_s = list(cv_folds(ms, test_start))
        rng = np.random.default_rng(C.SEED)
        trials = [dict(M.DEFAULT_PARAMS)] + [search_space(rng) for _ in range(N_TRIALS)]
        best, best_mae, log = None, np.inf, []
        for i, params in enumerate(trials):
            with mlflow.start_run(run_name=f"trial-{i}", nested=True):
                maes, _ = cv_score(Xs, ms, folds_s, params)
                mlflow.log_params(params)
                mlflow.log_metrics({"cv_mae_mean": float(np.mean(maes)),
                                    "cv_mae_std": float(np.std(maes))})
            log.append({"trial": i, **params, "cv_mae": float(np.mean(maes)), "folds": maes})
            print(f"trial {i}: cv MAE {np.mean(maes):.3f}  {params}")
            if np.mean(maes) < best_mae:
                best, best_mae = params, float(np.mean(maes))
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_cv_mae", best_mae)

        # ---- 2. out-of-fold quantile predictions -> CQR offsets ----
        q_maes, oof = cv_score(X, meta, folds, best, quantiles=C.QUANTILES)
        offsets = cqr_offsets(oof)
        oof_cov = E.interval_metrics(oof.y, oof.q10, oof.q90)
        cal_cov = E.interval_metrics(oof.y, *(M.apply_cqr(oof, oof.horizon.to_numpy(), offsets)[c]
                                             for c in ("q10", "q90")))
        print("OOF coverage raw", oof_cov, "after CQR", cal_cov)

        # ---- 3. hold-out evaluation ----
        tr = (meta["target_time"] < test_start).to_numpy()
        te = (meta["origin"] >= test_start).to_numpy()
        forecaster = M.Forecaster(params=best).fit(X[tr], meta.y[tr])
        pred_raw = forecaster.predict(X[te])
        mt = meta[te].reset_index(drop=True)
        pred = M.apply_cqr(pred_raw, mt.horizon.to_numpy(), offsets)

        mt_out = pd.concat([mt, pred.add_prefix("lgbm_")], axis=1)
        mt_out["lgbm_q10_raw"], mt_out["lgbm_q90_raw"] = pred_raw["q10"], pred_raw["q90"]

        # Holt-Winters on a shared random sample of test origins
        origins = np.sort(rng.choice(mt.origin.unique(), size=HW_ORIGINS, replace=False))
        hw_rows = []
        for o in pd.to_datetime(origins):
            fc = M.holt_winters_forecast(df["pm2_5"].loc[:o])
            hw_rows.append(pd.DataFrame({"origin": o, "horizon": np.arange(1, C.HORIZON + 1),
                                         "hw": fc}))
        mt_out = mt_out.merge(pd.concat(hw_rows), on=["origin", "horizon"], how="left")
        mt_out.to_parquet(C.PROCESSED_DIR / "test_predictions.parquet")

        def block(sub_df: pd.DataFrame, col: str) -> dict:
            return E.point_metrics(sub_df["y"], sub_df[col])

        methods = {"persistence": "persistence", "seasonal_naive_1d": "snaive",
                   "seasonal_naive_3d_mean": "snaive_3d", "lgbm": "lgbm_q50"}
        holdout = {name: block(mt_out, col) for name, col in methods.items()}
        shared = mt_out.dropna(subset=["hw"])
        shared_cmp = {name: block(shared, col) for name, col in methods.items()}
        shared_cmp["holt_winters"] = block(shared, "hw")

        by_h = {name: E.by_horizon(mt_out, mt_out[col]) for name, col in methods.items()}
        by_h["holt_winters"] = E.by_horizon(shared, shared["hw"])
        season = mt_out["target_time"].dt.month.isin([11, 12, 1, 2])
        seasons = {"smog_season_nov_feb": block(mt_out[season], "lgbm_q50"),
                   "rest_of_year": block(mt_out[~season], "lgbm_q50"),
                   "snaive_smog_season": block(mt_out[season], "snaive_3d"),
                   "snaive_rest_of_year": block(mt_out[~season], "snaive_3d")}
        alerts = {"lgbm": E.alert_metrics(mt_out.y, mt_out.lgbm_q50),
                  "seasonal_naive_3d_mean": E.alert_metrics(mt_out.y, mt_out.snaive_3d)}
        alerts_by_h = {n: E.alert_metrics(g.y, g.lgbm_q50)
                       for n, g in {k: mt_out[(mt_out.horizon >= lo) & (mt_out.horizon <= hi)]
                                    for k, (lo, hi) in E.HORIZON_BUCKETS.items()}.items()}
        interval = {"raw_80pct_band": E.interval_metrics(mt_out.y, mt_out.lgbm_q10_raw,
                                                         mt_out.lgbm_q90_raw),
                    "cqr_80pct_band": E.interval_metrics(mt_out.y, mt_out.lgbm_q10, mt_out.lgbm_q90)}

        metrics = {
            "data": {"rows_hourly": len(df), "start": str(df.index.min()), "end": str(df.index.max()),
                     "holdout_start": str(test_start), "train_rows_supervised": int(tr.sum()),
                     "holdout_rows_supervised": int(te.sum())},
            "best_params": best,
            "cv": {"best_mae_mean": best_mae, "trials": log,
                   "oof_interval_raw": oof_cov, "oof_interval_cqr": cal_cov},
            "holdout": holdout,
            "holdout_shared_sample": {"origins": HW_ORIGINS, **shared_cmp},
            "by_horizon": by_h,
            "seasons": seasons,
            "alerts": alerts,
            "alerts_by_horizon": alerts_by_h,
            "interval": interval,
            "cqr_offsets": offsets,
        }
        (C.ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1))
        mlflow.log_metrics({
            "holdout_mae": holdout["lgbm"]["mae"], "holdout_rmse": holdout["lgbm"]["rmse"],
            "holdout_r2": holdout["lgbm"]["r2"], "holdout_alert_f1": alerts["lgbm"]["alert_f1"],
            "holdout_alert_recall": alerts["lgbm"]["alert_recall"],
            "holdout_snaive_mae": holdout["seasonal_naive_3d_mean"]["mae"],
            "band_coverage_pct": interval["cqr_80pct_band"]["coverage_pct"],
        })
        forecaster.save(C.MODELS_DIR / "holdout")  # kept locally (gitignored) for report.py

        # ---- 4. production refit on ALL data ----
        final = M.Forecaster(params=best).fit(X, meta.y)
        final.save(C.ARTIFACTS_DIR / "model")
        (C.ARTIFACTS_DIR / "model" / "cqr_offsets.json").write_text(json.dumps(offsets))
        mlflow.log_artifacts(str(C.ARTIFACTS_DIR), artifact_path="artifacts")
        print(f"run {parent.info.run_id} done in {(time.time() - t_start) / 60:.1f} min")
        print(json.dumps({k: metrics[k] for k in ("holdout", "holdout_shared_sample", "interval")},
                         indent=1))


if __name__ == "__main__":
    main()
