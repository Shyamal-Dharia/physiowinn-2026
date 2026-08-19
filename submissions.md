# Submission Ledger

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
