
## 2026-09-19 ~05:30 UTC — FIRST QUALIFIED DEPLOY (C6)
- 40-seed held-out gate (seeds 2580-2639, never screened on): **C6_blind_noreserve_noevade +18.1%**
  (W27/L13 = 67.5% wins, floor p10 +1,977, variance 3.35M vs BASE 4.69M); **C3_no_reserve +13.7%**
  (W26/L14, floor +1,322). Both cleared the three deploy rules.
- Note the lesson: C6 scored +1.0% on the 20-seed screen and +18.1% at 40 seeds. The screen is very
  noisy in BOTH directions — the holdout rule earned its keep for the second time in one night.
- DEPLOYED C6 to the serving box: params-only change (controller hash f90cb4e3 unchanged), verified
  from inside the running container, /predict 1.2-1.9 ms. Rollback file: experiments/ROLLBACK_params_f90cb4e3.json
  and /root/params_backup_f90cb4e3.json on the serving box. Standby updated to match and left stopped.
- C6 = deletions only: evade_mode 0 (closed E3 line), reserve_frac 0 (refuted), blind_explore_frac 0.12
  and wander_weight 0.03 (less blind wandering).
- NEXT: user runs validations; then Q2 (scale the counterfactual oracle) and Q4 (bet B tier-2).
