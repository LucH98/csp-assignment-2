# Replication of Scheffer et al. (2009): early-warning signals

Harvested-population model: dX/dt = X(1 - X/K) - c X^2/(X^2 + 1), with K = 10.

`replication.py` makes two figures:

1. `figures/fold_bifurcation.png`: equilibria, fold points F1 and F2, and hysteresis
2. `figures/early_warning.png`: simulated biomass with slowly increasing c, and the
   rising standard deviation and lag-1 autocorrelation before the transition

## RUNME

```bash
pip install -r requirements.txt
python replication.py
```

## Settings (our choices; not reported in the paper)

| Setting | Value |
|---|---|
| Random seed | 42 |
| Noise intensity | 0.1 |
| Harvest rate | 1.0 -> 2.8, linearly over 1000 time units |
| Observations | one per time unit |
| Detrending | centred moving average, 51 points |
| Moving window | half of the pre-transition series |

## Results

| Check | Value |
|---|---|
| F1 (recovery threshold) | c = 1.787 |
| F2 (collapse threshold) | c = 2.604 |
| Kendall tau s.d. / AR(1) | 0.84 / 0.79 |
