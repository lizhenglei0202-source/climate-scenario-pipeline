# Climate Scenario Extraction Pipeline

> Multi-agent LLM pipeline that extracts scenario-to-output pairs from the peer-reviewed
> climate literature, released with the dataset and audit tables behind our manuscript
> (under review) on how that evidence base compares with the AR6 Scenarios Database and
> with national net-zero pledges.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

---

## The dataset in one paragraph

`data/climate_scenarios_dataset.xlsx` holds **42828 candidate scenario-output pairs**, one row
per candidate, carrying the full three-model debate trail. Of these, **22404 entered the
two-round debate** and **19502 passed it**, drawn from **10789 of 32726 screened papers**.
**13514** of the validated pairs carry an explicit AR6 C1 to C8 label.

Filter the released table exactly as the analysis does:

```python
import pandas as pd
df = pd.read_excel("data/climate_scenarios_dataset.xlsx")
outcome = df["Debate Outcome"].fillna("")
valid = df[outcome.str.contains("VALID", case=False)
           & ~outcome.str.contains("INVALID", case=False)]   # 19502 rows
```

The `~contains("INVALID")` half is **not optional**. `"INVALID SCENARIO"` contains `"VALID"` as
a substring, so a containment test alone admits the 2902 rejected rows and inflates every
downstream count. An earlier release of this README reported those inflated figures.

---

## Layout

```
pipeline/     the extraction pipeline that produced the dataset
  harvest_papers.py        harvesting from WoS, OpenAlex, PubMed, Semantic Scholar, arXiv
  extract_scenarios.py     13 expert prompts, three-model debate, retrieval-grounded auto-fix
  export_results.py        per-paper JSON to the flat released table

data/
  climate_scenarios_dataset.xlsx   the released table
  analysis_outputs/                derived and audit tables behind each figure and number
  policy/                          national pledges and the audited sector-plan screen
```

## Derived tables

Analysis and figure-drawing code is not part of this release. Every derived and audit
table the manuscript relies on is shipped ready-made under `data/analysis_outputs/` and
`data/policy/`, and the filter snippet above reproduces the validated subset the analysis
starts from. `publication_year_overrides_0731.csv` records the manually verified
first-online-year corrections used by the temporal results.

## Running the extraction pipeline

The pipeline calls three commercial APIs. **No key of ours is stored in this repository**, and
none is needed to reproduce the numbers, only to re-run extraction from PDFs.

Copy the template and fill in your own values:

```bash
cp .env.example .env
$EDITOR .env          # paste your keys, set PDF_CORPUS_DIR
python pipeline/extract_scenarios.py
```

Or export the same names as environment variables instead of using a file. The three keys and
the PDF directory are required; every other setting in `.env.example` has a working default.

| Setting | What it is for |
|---|---|
| `OPENAI_API_KEY` | extraction, screening, auto-fix, and debate panel member 1 |
| `GEMINI_API_KEY` | debate panel member 2 |
| `KIMI_API_KEY` | debate panel member 3 |
| `PDF_CORPUS_DIR` | any directory of PDFs, e.g. a Zotero storage folder or `harvest_papers.py` output |

If a required setting is missing the script stops immediately and names it, rather than
failing later with an authentication error. `.env` is git-ignored; `.env.example` is not.

Per PDF the pipeline runs title-and-abstract screening, routing to one of 13 IPCC-aligned
expert prompts, paragraph-level retrieval-augmented extraction of five fields per candidate,
a two-round debate across the three models, and a retrieval-grounded auto-fix that regenerates
only the fields the panel flagged.

Some expert prompts inside `pipeline/` are written in Chinese. They are functional inputs
rather than comments, and translating them would change extraction behaviour, so they are
released as run.

## Policy data

`data/policy/` carries the top-down side of the comparison. `national_pledges.csv` audits the
63 entries in the earlier compilation and marks the 61 current targets retained after the
July 2026 status review. Target selection prioritises law over a policy document, declaration
or pledge, and proposed/discussion status. `own_pledge_shares.csv` reports the full 33-economy
Custom-family comparison underlying Supplementary Fig. S7c; `data/analysis_outputs/custom_netzero_by_country.csv`
adds the corresponding unique-file counts used in Supplementary Table S14. For the sector screen (Supplementary Fig. S8),
`sector_plan_directional_screen.csv` contains the 37 economy–sector records retained from the
99-record legacy workbook after requiring a current quantified 2030 milestone and at least one
interpretable comparison with the selected sector reference;
`sector_plan_exclusion_log.csv` records why the other 62 were omitted, and
`sector_plan_benchmarks.csv` preserves the published benchmark definitions and sources. The
three directional outcomes are an analyst-coded screen, not IPCC C1–C8 categories, modeled
warming outcomes, or ratings produced by IEA, IPCC or the Climate Action Tracker.

## Known limitations

- Country mapping resolves free-text study locations against a Natural Earth alias table.
  Regional strings that collide with real ISO3 codes need explicit handling; `north america`
  collides with Namibia's `NAM` and is filtered before mapping.
- Net-zero years stated as never reaching net zero are stored as a 2110 sentinel and must be
  excluded before any median is taken.
- Carbon budgets are retained only when a value is explicitly tied to a carbon-mass unit or is
  the sole scalar/range in the dedicated budget field. The loose AR6-category cross-check is
  released as an audit flag and does not delete observations from Supplementary Fig. S7a.
- `Study Location` identifies the geography of an output, not ownership of the accompanying
  pathway. The manuscript's timing comparison (Figure 3) is therefore an evidence-policy comparison rather than
  a national compliance estimate.
- Recall is not measured. Every metric describes the quality of extracted pairs, not the
  fraction of in-corpus pairs that were found.

## Citation

Manuscript under review. Please cite this repository until the paper appears.

## License

MIT. See `LICENSE`.
