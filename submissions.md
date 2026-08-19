# Submission Ledger

## eeg-focal-ensemble-oof-0704

- commit: `ae9cfe6`
- branch: `official-refresh`
- internal: age-conditioned AUROC `0.7037`, reward `0.3080` (3-fold OOF, snapshot-averaged, no checkpoint selection)
- notes: EEG only. Six models -- window CNN, MIL-transformer with symmetric focal
  (gamma=2), MIL-transformer with asymmetric focal (gamma_pos=0, gamma_neg=2) -- two
  seeds each, snapshot-averaged over all 10 epochs, per-member empirical-CDF
  calibration, equal-weight blend, binary threshold at rank 0.75. No CAISR / no
  annotation dependency. Supersedes `d54c911` (0.6882 / 0.2601).

## eeg-only-snapshot-ensemble-oof-0688

- commit: `d54c911`
- branch: `official-refresh`
- internal: age-conditioned AUROC `0.688`, reward `0.264` (3-fold OOF)
- notes: EEG-only CNN x3 seeds, snapshot averaging, channel dropout, tolerant inference,
  rank-0.75 binary threshold. Superseded by the focal ensemble above.

## caisr-blend-leaderboard-auroc-0599

- commit: `12fc671`
- branch: `official-refresh`
- official age-conditioned AUROC: `0.599`
- notes: CAISR tabular 67.5% + CNN/ResCNN 32.5%. Scored 0.821 on random-split OOF and 0.599
  officially. The BMI-missingness curation artifact inflates random-split CV and does not transfer;
  the tabular branch was dropped afterwards.

## lejepa-calibrated-leaderboard-auroc-0669-reward-0049

- commit: `9289f5d8`
- branch: `official-refresh`
- official reward: `0.049`
- official age-conditioned AUROC: `0.669`
- notes: Calibrated joint EEG CNN-JEPA submission.

## cnn-leaderboard-auroc-0748-reward-0004

- commit: `ef915235`
- branch: `submission-cnn-2026-07-02`
- official reward: `0.004`
- official age-conditioned AUROC: `0.748`
- notes: CNN EEG submission before LeJEPA work.
