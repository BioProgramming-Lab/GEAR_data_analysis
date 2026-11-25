# Final statistical figures

`scripts/analyze_final.py` reads the per-TIF annotation CSVs and writes one
composition figure per experiment:

- `figures/fig_osmolarity_composition.{svg,pdf,txt}`
- `figures/fig_ficoll_composition.{svg,pdf,txt}`
- `figures/fig_dna_composition.{svg,pdf,txt}`
- `figures/fig_phase_composition.{svg,pdf,txt}`

Each figure shows the attached vs no-filament composition by diameter bin for
the conditions in that experiment. The final size gates are **<15, 15–20,
20–25, and ≥25 µm**; the `≥25` bin avoids the underpowered standalone `>30`
category. `unsure` objects are retained in the
annotation files but are not included in the attached/no-filament composition.
`not_guv` objects are absent from the annotations. The final annotations
contain 54 fields..

Each figure has a companion `.txt` file with the per-condition, per-size-bin
counts, the attached/none classification counts, `n_fields`, and the attached
ratio.

Run:

```bash
python scripts/analyze_final.py
```
