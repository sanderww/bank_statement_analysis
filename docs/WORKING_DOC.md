# Working document — main project tracker

**This is the main doc for tracking this project** (plan, task tables with
statuses, decisions, log). Superseded docs live in `docs/archived/`.

Originally the tracker for porting the good ideas from the **prototype** into
**V1** (this repo). The prototype is a reference to mine for features and
improvements. Tick items off (`[x]`) only when verified done. Spec: `GOAL.MD`
at the workspace root. Detailed findings + decisions below.

- **Keep / ship:** `bank_statement_analysis_v1/` (`src/` layout, FastAPI +
  `static/index.html` UI on `:8000`, git remote `sanderww/bank_statement_analysis`, branch `dev`)
- **Reference / mine for ideas:** `bank_statements_prototpye/` (Streamlit,
  SQLite-backed; not a git repo)

**Dev + sync workflow:** All changes are made **on this machine**, which has **no
real bank statements** (use synthetic/reference data only). Work is transferred to
the **other Mac** — which holds the actual statements and real trained models — via
**git bundle**. Keep commits clean and self-contained. V1's GitHub remote is
**public** — never commit real data, and gitignore anything derived from it.

---

## Assumption check (2026-07-04) — findings

1. **V1 git**: clean, on `dev`, in sync with `origin/dev`. ✅
2. **Privacy gap (public repo)** ⚠️: `bank_statements/` is NOT gitignored, and
   `models/transactions_classifier.joblib` (trained on real data — its TF-IDF
   vocabulary embeds real transaction descriptions) IS committed and public.
   Fix: gitignore both, `git rm --cached` the artefact. **User decision needed
   later**: the artefact remains in git history unless history is rewritten.
3. **Category-set bug**: `prompts/v2.txt` uses categories 0–10 (0 Unknown,
   10 Income) but `categorize.py` enum + OpenAI JSON schema only allow 1–9.
   Local model + training share the same 1–9 limit. Adopt the prototype's 0–10.
4. **Extraction**: prototype's `extract.py` was adapted *from* V1 — regex logic
   is identical. The prototype's genuine backend improvements are around
   extraction: normalisation (abs amount, drop empty/zero rows), **direction
   inference** (balance-delta + keyword fallback), **dedup** (date|desc|amount|balance
   key), signed amounts, and **confidence** from `predict_proba`.
5. **V1 has no real tests**: `src/tests/test_extract.py` is a stray reference
   script (with `__main__`), not pytest. pytest isn't a dependency.
6. **V1 paths are hardcoded** relative to `__file__` in several modules — needs
   a `config.py` with env overrides so tests can run against tmp dirs.
7. No `OPENAI_API_KEY` needed for anything except the OpenAI categorise path
   (untestable on this machine — fine; it stays as-is functionally).

## Design decisions (locked for this iteration)

- **Stay file-based** (no SQLite). V1's explicit folders are its UX strength and
  keep data inspectable. App state lives in `settings.json` (gitignored):
  active prompt version, active model version, confidence threshold.
- **Category set 0–10** (0 Unknown, 1–9 expenses, 10 Income) in a new
  `categories.py` — single source of truth for prompt/schema/model/UI.
- **Pipeline enrichment**: after extraction, rows are normalised → direction +
  signed_amount derived → deduped. CSV schema becomes
  `date, description, amount, balance, direction, signed_amount[, category,
  category_label, source, confidence]`.
- **Versioned prompts**: keep `prompts/v{N}.txt` files; API lists/activates/creates
  versions; default active = v2 (per `docs/Next_steps.md`).
- **Versioned models**: `models/v{N}/model.joblib` + `metadata.json` (training
  size, metrics incl. holdout accuracy, feature set, sources, created_at).
  Active version chosen in settings. Old single-file artefact path retired.
- **Review**: server endpoints to load/save a categorised CSV as JSON; UI table
  with inline category dropdowns, text/category/low-confidence filters,
  bulk accept. Edits mark `source=user`.
- **Training data managed separately** (goal 4): a reviewed statement is
  explicitly promoted ("add to training data") into `models/training_data/`
  (gitignored), deduped on the dedup key. Models train from that curated set —
  never silently from raw categorised output. This is the fine-tune-over-time loop:
  extract → categorise (LLM/local) → review/fix → promote to training data →
  train new model version → better local categorisation next time.
- **Charts (goal 5, keep minimal)**: one Insights section, Chart.js via CDN,
  exactly three: income vs costs, costs by category (donut), costs per month
  (stacked bar). Fed by reviewed/categorised CSVs.
- **Tests**: pytest as dev dependency; synthetic fixtures only (real-ish FNB-style
  descriptions invented). No real data in the V1 repo, ever.

---

## Implementation plan (2026-07-04) — status

Executing top to bottom; each task = one commit on `dev`.

| # | Task | Status |
|---|------|--------|
| 1 | **Hygiene**: .gitignore `bank_statements/`, `/models/`, `settings.json`, `output/`; `git rm --cached` the committed model artefact; remove stray `src/tests/test_extract.py` script | done (04ae264) |
| 2 | **Test scaffolding + config.py**: central paths w/ env overrides (`BSA_PROJECT_DATA`, `BSA_PROMPTS_DIR`), refactor existing modules to use it, pytest dev-dep, conftest with tmp dirs, first smoke tests | done (e7b2494) |
| 3 | **categories.py (0–10)** + refactor `categorize.py` (enum/schema/labels) + `train_model.py`; activate prompt v2 as default | done (b3fee05) |
| 4 | **direction.py + dedup.py + normalise** ported from prototype, integrated into `services.extract_data`; extended CSV columns; unit tests | done (dbafad1) |
| 5 | **settings.py + prompt_store.py + model_store.py**: versioned prompts/models, active selection, metadata + holdout metrics; local categorisation gains `confidence` (predict_proba); tests | done (d8de501) |
| 6 | **API**: `/api/settings`, `/api/prompts`, `/api/models` (+ train), `/api/review/*` (load/save/promote-to-training), `/api/insights` aggregates; TestClient tests | done (0bc1bce) |
| 7 | **UI**: Review step (inline edit, filters, low-conf highlight), Insights (metrics + 2 charts), Setup panel (prompts/models/settings/training data); keep V1 look & feel | done (652b001) |
| 8 | **Docs**: README + Makefile updated to new workflow; training-data loop documented | done (dbcd3c8) |

**All 8 tasks done — 42 tests pass; verified end-to-end against the live server
with synthetic data (extract CSV → local categorise with confidence → review →
promote → train v1 → insights). Ready for user testing at http://127.0.0.1:8000.**

### Goal-by-goal outcome (vs GOAL.MD)

1. **Extraction**: prototype's regexes were identical to V1's (adapted from it);
   what V1 lacked was around extraction — now ported: normalisation, direction
   inference (balance delta + keyword fallback), dedup, signed amounts.
2. **Settings/prompts/models**: versioned prompts (`prompts/v{N}.txt`) and
   models (`models/v{N}/` + metadata) with activate/create/train in the Setup
   section; thresholds + OpenAI model in settings.json.
3. **Review**: Review & Improve section — inline category dropdowns, search /
   category / low-confidence filters, amber low-confidence highlighting, save,
   promote to training data. Edits are marked `source=user`.
4. **Sense check — training data managed separately**: yes, by construction now.
   Models train ONLY from `models/training_data/` (curated, grown explicitly via
   promote; deduped on the transaction key). Metadata records what each version
   was trained on; holdout accuracy is reported when there's enough data.
5. **Charts**: kept minimal — metric cards + costs-by-category doughnut +
   income-vs-costs-per-month bar, on selectable categorised files.

### Task details / notes

- T2 env override prefix: `BSA_` (e.g. `BSA_PROJECT_DATA` pointing at an alt root
  for statements/output/models/prompts/settings) so tests + other-Mac layout can
  relocate data without code edits.
- T4 direction port: balance-delta primary signal (tolerance 0.05), keyword
  fallback (salar/credit/refund/reversal/deposit/interest/cashback/inward),
  default `out`. Dedup drops within-batch duplicates at extract time (overlapping
  statements) — matches prototype semantics minus the DB.
- T5 model metadata JSON per version; `train` computes stratified holdout accuracy
  when ≥20 rows & ≥2/class (port of prototype `_fit_and_score`).
- T6 review save rewrites the categorised CSV in place; changed rows get
  `source=user`. Promote-to-training appends deduped rows to
  `models/training_data/training_data.csv`.
- T7 charts contract mirrors prototype `charts.py` aggregation (pure functions
  server-side, Chart.js render client-side).

---

## Iteration 2 (2026-07-04) — styling + remaining backlog

| # | Task | Status |
|---|------|--------|
| 9 | **Backend**: `source_statement` traceability per row; `handoff.py` (Claude Code file round-trip, no API) + API; decoupled final export (`output/final/`) + API; tests | done (fc2a75a) |
| 10 | **Restyle UI** — distinctive "ledger/print" look (Fraunces serif + IBM Plex Mono, paper background, ink borders + offset shadows, banker's-green accent, ruled tables) — awaiting user verdict | done (76c334b) |
| 11 | **UI features**: Claude hand-off block in step 2; "Export final CSV" in review; review-table pagination (100/page) | done (76c334b) |
| 12 | **Prompt editing** (user request): edit in place (`PUT /api/prompts/{v}`) + save-as-new-version, in the Prompts tab | done (fc2a75a + 76c334b) |

49 tests green; hand-off round trip, final export and prompt editing verified
live against the running server. Server now runs with `--reload`.

## Syncing between machines (git bundle)

This repo is synced Mac-to-Mac by copying a git bundle (no push to the shared
remote needed). Full bundles are idempotent and safest.

**Create (source Mac):**
```bash
git bundle create ../bank_statement_analysis-$(date +%Y-%m-%d).bundle --all
git bundle verify ../bank_statement_analysis-<date>.bundle
```

**Apply (receiving Mac):**
```bash
git bundle verify /path/to/bank_statement_analysis-<date>.bundle
git fetch /path/to/bank_statement_analysis-<date>.bundle dev:refs/remotes/bundle/dev
git checkout dev
git merge bundle/dev        # normal case (fast-forward)
# OR, if history was rewritten on the source (as on 2026-07-04, authors reset):
git reset --hard bundle/dev # stash/commit local changes first!
```

**One-off after the 2026-07-05 bundle** (history was rewritten): use the
`reset --hard` variant, then retrain the local model — the old single-file
artefact and 1–9 category labels are retired. Flow: categorise a statement →
review → "Add to training data" → Setup → Train.

## Backlog (not doable on this Mac / user decision)

- Git history rewrite to purge the committed real-data model artefact (user
  call — needs `git filter-repo` + force push).
- Validate direction inference + description+amount features on real statements
  (other Mac only).

---

## Log

- **2026-07-05** — Docs reorganised: this doc moved into the repo as the main
  tracker; superseded docs → `docs/archived/`. **Redacted real statement data**
  (name, address, account number, transactions) found in the old project brief —
  original remains in git history; history rewrite now covers TWO leaks (model
  artefact + project.md). Restyled UI to match budget_calculator (388ae99).
  Remote push postponed (no sanderww credential on this machine) — syncing to
  the other Mac via git bundle instead; commits re-authored as sanderww.

- **2026-07-04** — Cloned prototype + V1 side by side; renamed; created this doc.
- **2026-07-04** — Both apps run locally: V1 `:8000` (uvicorn), prototype `:8501`
  (streamlit). Prototype `.venv` had stale shebangs after rename → launch via
  `.venv/bin/python -m streamlit`.
- **2026-07-04** — Read both codebases fully; assumption check done (findings
  above); design decisions locked; plan drawn up. Executing.
- **2026-07-04** — Commit authors rewritten from the work email to
  `sanderww <30800089+sanderww@users.noreply.github.com>` before first push
  (public personal repo); repo-local git identity set. Hashes changed accordingly.
- **2026-07-04** — Iteration 2 (fc2a75a, 76c334b): ledger restyle (user to judge),
  Claude Code hand-off, decoupled final export (`output/final/`),
  `source_statement` traceability, review pagination, prompt editing.
  Remaining backlog is other-Mac / user-decision only.
- **2026-07-04** — User feedback: Setup shouldn't sit at the bottom of the page.
  Moved into a slide-over settings drawer opened via a prominent ⚙ Setup button
  in the header, with tabs General / Prompts / Models & Training, in-drawer
  status messages, Escape/backdrop close (f1bdb27).
- **2026-07-04** — Executed tasks 1–8 (commits 04ae264 → dbcd3c8 on `dev`).
  42 tests green. Synthetic demo data placed in `output/extracted_raw/` and
  `output/categorised/` (gitignored) so the UI is testable without real
  statements; a demo model v1 was trained from it. Server running on :8000.
  Open decision for user: purge the old real-data model artefact from git
  history (needs `git filter-repo` + force push).
