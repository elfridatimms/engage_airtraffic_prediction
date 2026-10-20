# Measured results

Selected models: classifier_engineered, ranker_base; final trees: {'classifier_engineered': 600, 'ranker_base': 600}.

Official validation (fixed CV-chosen tree counts): top-1 35.6230%, top-2 51.9968%, **top-3 61.4217%** (1252 snapshots).
Controller CV: selected mean fold top-3 **64.0176%**. Pooled OOF top-3 64.0191%; equal-controller mean 64.5837%.

## Controller CV

```
                model  mean_top3  std_top3  worst_fold  total_samples
classifier_engineered   0.636587  0.005859    0.630246          11923
  classifier_weighted   0.635125  0.009381    0.624360          11923
      classifier_time   0.635079  0.003457    0.631781          11923
    classifier_depth8   0.631619  0.005518    0.625640          11923
          ranker_base   0.631330  0.007246    0.623593          11923
  classifier_identity   0.625162  0.002458    0.622601          11923
      classifier_base   0.625062  0.001328    0.623603          11923
    ranker_engineered   0.620829  0.002575    0.619243          11923
```

## Official validation experiments

```
                model     top1     top2     top3  trees
      classifier_base 0.342652 0.514377 0.623003    500
          ranker_base 0.359425 0.535942 0.639776    500
classifier_engineered 0.354633 0.512780 0.614217    200
  classifier_weighted 0.364217 0.518371 0.629393    400
    ranker_engineered 0.365815 0.523962 0.615815    200
  classifier_identity 0.327476 0.510383 0.643770    550
      classifier_time 0.353035 0.512780 0.619010    250
    classifier_depth8 0.345048 0.511981 0.640575    579
```

## Selected official-validation sessions

```
dimension  value  samples     top1     top2     top3  rank4
  session 019sc1      378 0.328042 0.455026 0.566138     29
  session 019sc2      475 0.326316 0.515789 0.597895     41
  session 020sc1      399 0.418546 0.586466 0.679198     40
```

## Task errors

```
dimension                    value  samples     top1     top2     top3  rank4
task_type          assume_aircraft      363 0.498623 0.724518 0.823691     23
task_type      conflict_resolution       39 0.128205 0.128205 0.179487     12
task_type       entry_coordination      128 0.000000 0.000000 0.000000     19
task_type          exit_conditions       41 0.024390 0.048780 0.048780      2
task_type exit_conflict_resolution       11 0.090909 0.181818 0.181818      0
task_type        exit_coordination       41 0.000000 0.000000 0.048780      4
task_type          qos_improvement       20 0.000000 0.050000 0.200000      6
task_type          return_to_route       56 0.017857 0.107143 0.321429     14
task_type        transfer_aircraft      382 0.554974 0.714660 0.806283     14
task_type            zone_conflict      171 0.263158 0.578947 0.742690     16
```

True task ranked exactly fourth: 110 / 1252 samples. Coordination-task recall should be read separately from frequent assume/transfer tasks; small task subsets are noisy.

## Feature interpretation

```
                          feature  importance
           abs_distance_to_sector   11.920216
               distance_to_sector   11.565496
 zone_conflict__zone_conflict_ind    5.625279
                       task_index    5.117555
                  l_rank_age_type    3.588639
k_timeToExitPoint_relative_median    3.249794
            x_task_age_percentile    3.127152
       x_task_age_relative_median    3.059090
               abs_time_to_sector    2.888439
                        task_twin    2.854357
                   time_to_sector    2.839916
                   a_since_flight    2.343784
                     log_task_age    2.127903
                      a_since_any    1.672146
            k_distanceToExitPoint    1.649815
```

Task type differentiates jobs associated with the same aircraft. List ranks describe relative urgency among competing candidates. Task age measures persistence; sector and exit timing describe impending entry or handover. Action history indicates recent controller attention. Missing-state patterns can distinguish aircraft not yet assumed. These associations explain model use, not causal effects.

## Scope and limitations

All scores above are measured locally, not Kaggle scores. Hyperparameters/checkpoints and ensemble choices use held-out controllers but share model-selection folds, so uncertainty includes selection optimism and correlated snapshots. No public-leaderboard feedback was used. Identity and clock features were tested explicitly; the selected configuration documents whether either was retained. Final training includes validation, so controller 020 is partially known while 023 remains unseen. No test labels, pseudo-labels, future target history, or label smoothing were used.

Original input hashes and final submission validation passed; submission contains exactly 1,863 samples. Component-only alternatives are retained in outputs; the primary submission follows controller-CV selection rather than the highest exploratory official score.
