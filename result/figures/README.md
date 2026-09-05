# Benchmark 결과 시각화

이 폴더의 그래프는 `result/fair_comparison_protocol_and_results.md`의 통합 결과를 시각화한다.
표에 기록된 값을 그림에 다시 입력하지 않고, 공개된 `outputs/`의 prediction CSV와
Regime 분석 CSV에서 계산한다.

## 그래프

- `backtest-equity-connected-oos.svg`: 동일한 비용과 24시간 non-overlap 규칙을 적용한
  누적 자산곡선
- `regime-macro-f1.svg`: 방향·변동성 Regime별 Macro F1
- `figure_metrics.json`: 그래프 생성에 사용한 재계산 결과

모델별 색상은 두 그래프에서 동일하다. 색상뿐 아니라 marker와 일부 line style도 함께
사용해 모델을 구분한다.

## 다시 생성하기

Repository root에서 다음 명령을 실행한다.

```bash
python src/evaluation/plot_benchmark_results.py --benchmark-root .
```

스크립트는 그림을 만들기 전에 계산 결과가 통합 결과표의 주요 수치와 일치하는지 확인한다.
모델 prediction이나 평가 결과가 바뀌었는데 문서의 값이 갱신되지 않았다면 오류를 발생시킨다.
