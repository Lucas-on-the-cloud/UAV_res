# Dataset Selection Notes

## Final choice: VisDrone-DET (person-filtered)

**Source:** https://github.com/VisDrone/VisDrone-Dataset
**Kaggle mirror used:** `banuprasadb/visdrone-dataset`
**License:** Research use

### Why VisDrone

- Open and immediately accessible (no email request required)
- Already in YOLO format on Kaggle mirror
- Large enough for training (5,684 train images after filtering)
- Aerial/drone perspective (not ground-level)
- Contains `pedestrian` and `people` classes relevant to SAR
- Inference patterns similar to SAR scenarios (small targets, varied backgrounds)

### Filtering

VisDrone has 10 object classes. We keep only person-related classes:

| Original class | New class | Train instances |
|---|---|---|
| 0: pedestrian | 0: person | 79,337 |
| 1: people | 0: person | 27,059 |
| 2-9: others | (discarded) | — |

**Filtered dataset:** 5,684 train images + 531 val images, ~106K person instances total.

Images without any person after filtering are removed.

### Limitations & how we address them

| Limitation | Mitigation |
|---|---|
| Urban scenes, not wilderness SAR | Cross-evaluate on HERIDAL/SARD in Phase 5 |
| Mostly daytime conditions | Augment with synthetic data covering varied weather/lighting in Phase 3 |
| Heavy class imbalance with other VisDrone classes | Resolved by filtering to person-only |

---

## Alternatives considered

| Dataset | Verdict | Reason |
|---|---|---|
| **HERIDAL** | Reserve for cross-eval | Email-gated access; adds latency |
| **SARD** | Backup for cross-eval | Search availability on Kaggle |
| **TinyPerson** | Not primary | Mostly maritime, less aligned with intended wilderness SAR |
| **AU-AIR** | Not primary | Traffic-focused, less SAR-relevant |
| **HIT-UAV** | Optional future work | Thermal imaging — different modality |
