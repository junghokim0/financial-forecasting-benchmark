#!/usr/bin/env python3
"""Generate reproducible SVG figures from the public benchmark outputs.

The script intentionally uses the same prediction artifacts and 24-hour
non-overlapping backtest implementation as the benchmark report.  It does not
retrain or modify any model.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .backtest import non_overlapping_backtest
except ImportError:  # Direct script execution.
    from backtest import non_overlapping_backtest


COLORS = {
    "Zero-return baseline": "#777777",
    "Ridge-Flat": "#0072B2",
    "LSTM": "#E69F00",
    "LSTM Classifier": "#E69F00",
    "TimesNet": "#009E73",
    "TimesNet Classifier": "#009E73",
    "Chronos-2 LoRA": "#CC79A7",
    "TimesFM 2.5 LoRA": "#D55E00",
    "Cryptova-Raw": "#56B4E9",
    "Cryptova-Base": "#6A3D9A",
    "Cryptova-Full": "#111111",
}

MARKERS = {
    "Zero-return baseline": "circle",
    "Ridge-Flat": "circle",
    "LSTM": "square",
    "LSTM Classifier": "square",
    "TimesNet": "triangle",
    "TimesNet Classifier": "triangle",
    "Chronos-2 LoRA": "diamond",
    "TimesFM 2.5 LoRA": "cross",
    "Cryptova-Raw": "circle",
    "Cryptova-Base": "square",
    "Cryptova-Full": "diamond",
}

DASHES = {
    "Cryptova-Raw": "8 4",
    "Cryptova-Base": "3 3",
    "Zero-return baseline": "5 4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing outputs/ and result/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <benchmark-root>/result/figures.",
    )
    return parser.parse_args()


def rolling_paths(root: Path, pattern: str) -> list[Path]:
    return [root / pattern.format(rolling=rolling) for rolling in (1, 2, 3)]


def load_predictions(paths: Iterable[Path]) -> pd.DataFrame:
    paths = list(paths)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prediction files:\n" + "\n".join(missing))
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    return frame.sort_values("sample_time").reset_index(drop=True)


def classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    recalls: list[float] = []
    f1s: list[float] = []
    for label in (0, 1, 2):
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "Macro F1": float(np.mean(f1s)),
        "Balanced Accuracy": float(np.mean(recalls)),
        "SHORT Recall": recalls[0],
        "HOLD Recall": recalls[1],
        "LONG Recall": recalls[2],
    }


def regression_metrics(frame: pd.DataFrame, zero_prediction: bool = False) -> dict[str, float]:
    actual = frame["raw_future_return"].to_numpy(dtype=float)
    predicted = (
        np.zeros_like(actual)
        if zero_prediction
        else frame["predicted_return"].to_numpy(dtype=float)
    )
    error = predicted - actual
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
    }


def regression_frames(root: Path) -> dict[str, pd.DataFrame]:
    ridge = load_predictions(
        rolling_paths(
            root,
            "outputs/ridge_flat/rolling_{rolling}/rmse_selected/predictions_test.csv",
        )
    )
    return {
        "Zero-return baseline": ridge,
        "Ridge-Flat": ridge,
        "LSTM": load_predictions(
            rolling_paths(
                root,
                "outputs/lstm_regression/rolling_{rolling}/rmse_selected/predictions_test.csv",
            )
        ),
        "TimesNet": load_predictions(
            [root / "outputs/timesnet_regression/connected_oos/predictions_test.csv"]
        ),
        "Chronos-2 LoRA": load_predictions(
            [root / "outputs/chronos2_lora_chart/connected_oos/predictions_test.csv"]
        ),
        "TimesFM 2.5 LoRA": load_predictions(
            [root / "outputs/timesfm2_5_lora_close/connected_oos/predictions_test.csv"]
        ),
    }


def classification_frames(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "Ridge-Flat": load_predictions(
            rolling_paths(
                root,
                "outputs/ridge_flat/rolling_{rolling}/macro_f1_selected/predictions_test.csv",
            )
        ),
        "LSTM Classifier": load_predictions(
            rolling_paths(
                root,
                "outputs/lstm_classifier/rolling_{rolling}/predictions_test.csv",
            )
        ),
        "TimesNet Classifier": load_predictions(
            [root / "outputs/timesnet_classifier/connected_oos/predictions_test.csv"]
        ),
        "Chronos-2 LoRA": load_predictions(
            [root / "outputs/chronos2_lora_chart/connected_oos/predictions_test.csv"]
        ),
        "TimesFM 2.5 LoRA": load_predictions(
            [root / "outputs/timesfm2_5_lora_close/connected_oos/predictions_test.csv"]
        ),
        "Cryptova-Raw": load_predictions(
            [root / "outputs/cryptova_raw_re_evaluation/connected_oos/predictions_test.csv"]
        ),
        "Cryptova-Base": load_predictions(
            [root / "outputs/cryptova_base_re_evaluation/connected_oos/predictions_test.csv"]
        ),
        "Cryptova-Full": load_predictions(
            [root / "outputs/cryptova_full_re_evaluation/connected_oos/predictions_test.csv"]
        ),
    }


def assert_close(name: str, actual: float, expected: float, tolerance: float = 5e-6) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise ValueError(f"{name}: calculated {actual:.9f}, expected {expected:.9f}")


def validate_against_report(
    regression: dict[str, dict[str, float]],
    classification: dict[str, dict[str, float]],
    backtests: dict[str, dict[str, float | int]],
) -> None:
    expected_rmse = {
        "Zero-return baseline": 0.022948,
        "Ridge-Flat": 0.023178,
        "LSTM": 0.023790,
        "TimesNet": 0.023916,
        "Chronos-2 LoRA": 0.024300,
        "TimesFM 2.5 LoRA": 0.024992,
    }
    expected_f1 = {
        "Ridge-Flat": 0.284691,
        "LSTM Classifier": 0.318197,
        "TimesNet Classifier": 0.364654,
        "Chronos-2 LoRA": 0.252000,
        "TimesFM 2.5 LoRA": 0.287013,
        "Cryptova-Full": 0.350898,
        "Cryptova-Base": 0.376506,
        "Cryptova-Raw": 0.381875,
    }
    expected_returns = {
        "Ridge-Flat": -0.00005,
        "LSTM Classifier": -0.3967,
        "TimesNet Classifier": -0.1621,
        "Chronos-2 LoRA": -0.2464,
        "TimesFM 2.5 LoRA": -0.3096,
        "Cryptova-Full": 0.2746,
        "Cryptova-Base": 0.0742,
        "Cryptova-Raw": -0.1811,
    }
    for model, expected in expected_rmse.items():
        assert_close(f"{model} RMSE", regression[model]["RMSE"], expected, 5e-6)
    for model, expected in expected_f1.items():
        assert_close(f"{model} Macro F1", classification[model]["Macro F1"], expected, 5e-6)
    for model, expected in expected_returns.items():
        assert_close(
            f"{model} cumulative return",
            float(backtests[model]["cumulative_return"]),
            expected,
            6e-5,
        )


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def marker_svg(kind: str, x: float, y: float, color: str, size: float = 5.0) -> str:
    if kind == "square":
        return f'<rect x="{x-size:.2f}" y="{y-size:.2f}" width="{2*size:.2f}" height="{2*size:.2f}" fill="{color}"/>'
    if kind == "triangle":
        points = f"{x:.2f},{y-size-1:.2f} {x-size-1:.2f},{y+size:.2f} {x+size+1:.2f},{y+size:.2f}"
        return f'<polygon points="{points}" fill="{color}"/>'
    if kind == "diamond":
        points = f"{x:.2f},{y-size-1:.2f} {x+size+1:.2f},{y:.2f} {x:.2f},{y+size+1:.2f} {x-size-1:.2f},{y:.2f}"
        return f'<polygon points="{points}" fill="{color}"/>'
    if kind == "cross":
        return (
            f'<path d="M {x-size:.2f} {y-size:.2f} L {x+size:.2f} {y+size:.2f} '
            f'M {x+size:.2f} {y-size:.2f} L {x-size:.2f} {y+size:.2f}" '
            f'stroke="{color}" stroke-width="2.4" fill="none"/>'
        )
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{color}"/>'


def svg_header(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title chart-desc">',
        f"<title id=\"chart-title\">{esc(title)}</title>",
        f"<desc id=\"chart-desc\">{esc(description)}</desc>",
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Arial,"Malgun Gothic",sans-serif;fill:#222} .title{font-size:24px;font-weight:700} .subtitle{font-size:14px;fill:#555} .axis{font-size:13px;fill:#333} .tick{font-size:12px;fill:#555} .legend{font-size:13px;fill:#222} .grid{stroke:#E4E7EB;stroke-width:1} .frame{stroke:#AEB5BD;stroke-width:1;fill:none}</style>',
    ]


def write_categorical_lines(
    output: Path,
    title: str,
    subtitle: str,
    categories: list[str],
    series: dict[str, list[float]],
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    y_label: str,
    value_format,
) -> None:
    width, height = 1280, 720
    left, top, bottom, legend_width = 92, 112, 92, 300
    plot_width = width - left - legend_width - 38
    plot_height = height - top - bottom
    right = left + plot_width
    bottom_y = top + plot_height
    x_positions = np.linspace(left + 24, right - 24, len(categories))

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = svg_header(width, height, title, subtitle)
    lines.extend(
        [
            f'<text class="title" x="{left}" y="42">{esc(title)}</text>',
            f'<text class="subtitle" x="{left}" y="70">{esc(subtitle)}</text>',
            f'<rect class="frame" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>',
        ]
    )
    for tick in y_ticks:
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        lines.append(f'<text class="tick" x="{left-12}" y="{y+4:.2f}" text-anchor="end">{esc(value_format(tick))}</text>')
    for x, category in zip(x_positions, categories):
        lines.append(f'<text class="axis" x="{x:.2f}" y="{bottom_y+30}" text-anchor="middle">{esc(category)}</text>')
    lines.append(f'<text class="axis" transform="translate(24,{top + plot_height/2:.2f}) rotate(-90)" text-anchor="middle">{esc(y_label)}</text>')

    for model, values in series.items():
        color = COLORS[model]
        points = [(float(x), y_pos(float(value))) for x, value in zip(x_positions, values)]
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash = f' stroke-dasharray="{DASHES[model]}"' if model in DASHES else ""
        lines.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.4"{dash}/>' )
        for x, y in points:
            lines.append(marker_svg(MARKERS[model], x, y, color))

    legend_x = right + 34
    legend_y = top + 8
    for index, model in enumerate(series):
        y = legend_y + index * 43
        color = COLORS[model]
        dash = f' stroke-dasharray="{DASHES[model]}"' if model in DASHES else ""
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+34}" y2="{y}" stroke="{color}" stroke-width="2.4"{dash}/>' )
        lines.append(marker_svg(MARKERS[model], legend_x + 17, y, color, 4.2))
        lines.append(f'<text class="legend" x="{legend_x+46}" y="{y+4}">{esc(model)}</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def write_equity_chart(
    output: Path,
    frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float | int]]:
    width, height = 1280, 720
    left, top, bottom, legend_width = 92, 112, 92, 300
    plot_width = width - left - legend_width - 38
    plot_height = height - top - bottom
    right = left + plot_width
    bottom_y = top + plot_height
    start = min(frame["sample_time"].min() for frame in frames.values())
    end = max(frame["target_time"].max() for frame in frames.values())
    curves: dict[str, tuple[list[pd.Timestamp], np.ndarray]] = {}
    backtests: dict[str, dict[str, float | int]] = {}
    all_wealth = [1.0]
    for model, frame in frames.items():
        metrics, trades = non_overlapping_backtest(frame)
        backtests[model] = metrics
        returns = trades["strategy_return"].to_numpy(dtype=float)
        wealth = np.cumprod(1.0 + returns)
        times = list(pd.to_datetime(trades["target_time"], utc=True))
        values = np.concatenate(([1.0], wealth, [wealth[-1] if len(wealth) else 1.0]))
        curve_times = [start, *times, end]
        curves[model] = (curve_times, values)
        all_wealth.extend(values.tolist())
    y_min = min(all_wealth)
    y_max = max(all_wealth)
    padding = max((y_max - y_min) * 0.10, 0.04)
    y_min = max(0.0, y_min - padding)
    y_max = y_max + padding

    def x_pos(value: pd.Timestamp) -> float:
        ratio = (value - start).total_seconds() / (end - start).total_seconds()
        return left + ratio * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    title = "거래비용 반영 Connected OOS 누적 자산곡선"
    subtitle = "24시간 non-overlap 거래 · 수익은 각 포지션의 청산 시점에 반영 · 시작 자산 = 1.0"
    lines = svg_header(
        width,
        height,
        title,
        "각 모델이 서로 다른 시점에 선택한 거래를 동일한 비용과 24시간 보유 규칙으로 평가한 누적 자산곡선",
    )
    lines.extend(
        [
            f'<text class="title" x="{left}" y="42">{title}</text>',
            f'<text class="subtitle" x="{left}" y="70">{subtitle}</text>',
            f'<rect class="frame" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>',
        ]
    )
    for tick in np.linspace(y_min, y_max, 6):
        y = y_pos(float(tick))
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        lines.append(f'<text class="tick" x="{left-12}" y="{y+4:.2f}" text-anchor="end">{tick:.2f}</text>')
    if y_min <= 1.0 <= y_max:
        y = y_pos(1.0)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#777" stroke-width="1.2" stroke-dasharray="4 4"/>')
    quarter_starts = pd.date_range(start=start.normalize(), end=end.normalize(), freq="QS")
    for timestamp in quarter_starts:
        if start <= timestamp <= end:
            x = x_pos(timestamp)
            lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom_y}"/>')
            lines.append(f'<text class="tick" x="{x:.2f}" y="{bottom_y+28}" text-anchor="middle">{timestamp.strftime("%Y-%m")}</text>')
    lines.append(f'<text class="axis" transform="translate(24,{top + plot_height/2:.2f}) rotate(-90)" text-anchor="middle">누적 자산</text>')
    lines.append(f'<text class="axis" x="{left + plot_width/2:.2f}" y="{height-22}" text-anchor="middle">청산 시점</text>')

    for model, (times, values) in curves.items():
        color = COLORS[model]
        points: list[tuple[float, float]] = []
        for index, (timestamp, value) in enumerate(zip(times, values)):
            x = x_pos(timestamp)
            y = y_pos(float(value))
            if index and points:
                points.append((x, points[-1][1]))
            points.append((x, y))
        path = " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))
        dash = f' stroke-dasharray="{DASHES[model]}"' if model in DASHES else ""
        lines.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>' )

    legend_x = right + 34
    legend_y = top + 8
    for index, model in enumerate(frames):
        y = legend_y + index * 43
        color = COLORS[model]
        dash = f' stroke-dasharray="{DASHES[model]}"' if model in DASHES else ""
        final_return = float(backtests[model]["cumulative_return"]) * 100.0
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+34}" y2="{y}" stroke="{color}" stroke-width="2.4"{dash}/>' )
        lines.append(marker_svg(MARKERS[model], legend_x + 17, y, color, 4.2))
        lines.append(f'<text class="legend" x="{legend_x+46}" y="{y+4}">{esc(model)}  {final_return:+.2f}%</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")
    return backtests


def main() -> None:
    args = parse_args()
    root = args.benchmark_root.resolve()
    output = (args.output_dir or root / "result" / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)

    reg_frames = regression_frames(root)
    cls_frames = classification_frames(root)
    regression = {
        model: regression_metrics(frame, zero_prediction=(model == "Zero-return baseline"))
        for model, frame in reg_frames.items()
    }
    classification = {
        model: classification_metrics(frame) for model, frame in cls_frames.items()
    }
    backtests = write_equity_chart(output / "backtest-equity-connected-oos.svg", cls_frames)
    validate_against_report(regression, classification, backtests)

    regression_categories = ["RMSE", "MAE"]
    write_categorical_lines(
        output / "regression-connected-oos.svg",
        "Connected OOS 수익률 예측 오차",
        "낮을수록 좋음 · 모든 값은 24시간 수익률 단위",
        regression_categories,
        {model: [metrics[key] for key in regression_categories] for model, metrics in regression.items()},
        0.0155,
        0.0255,
        [0.016, 0.018, 0.020, 0.022, 0.024],
        "오차",
        lambda value: f"{value:.3f}",
    )

    classification_keys = [
        "Macro F1",
        "Balanced Accuracy",
        "SHORT Recall",
        "HOLD Recall",
        "LONG Recall",
    ]
    classification_categories = [
        "Macro F1",
        "Balanced Acc.",
        "SHORT Recall",
        "HOLD Recall",
        "LONG Recall",
    ]
    write_categorical_lines(
        output / "classification-connected-oos.svg",
        "Connected OOS 신호 분류 성능",
        "동일한 6,291개 시간별 표본 · 미래 24시간 SHORT / HOLD / LONG",
        classification_categories,
        {
            model: [metrics[key] for key in classification_keys]
            for model, metrics in classification.items()
        },
        0.0,
        1.0,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "점수",
        lambda value: f"{value:.1f}",
    )

    regime_path = root / "outputs" / "regime_analysis" / "classification_by_regime.csv"
    regime = pd.read_csv(regime_path)
    regime_order = ["UP", "DOWN", "SIDEWAYS", "HIGH", "LOW"]
    regime_labels = ["단기 상승", "단기 하락", "단기 중립", "고변동성", "저변동성"]
    regime_series: dict[str, list[float]] = {}
    for model in cls_frames:
        model_rows = regime[regime["model"] == model].set_index("regime")
        regime_series[model] = [float(model_rows.loc[name, "macro_f1"]) for name in regime_order]
    write_categorical_lines(
        output / "regime-macro-f1.svg",
        "시장 Regime별 신호 분류 성능",
        "Connected OOS Macro F1 · Regime은 진입 전 과거 정보만으로 정의",
        regime_labels,
        regime_series,
        0.20,
        0.45,
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        "Macro F1",
        lambda value: f"{value:.2f}",
    )

    summary = {
        "source": "public prediction artifacts and regime_analysis CSV",
        "regression": regression,
        "classification": classification,
        "backtest": backtests,
        "figures": sorted(path.name for path in output.glob("*.svg")),
    }
    (output / "figure_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "figures": summary["figures"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
