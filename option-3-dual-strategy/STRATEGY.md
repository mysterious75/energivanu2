# Option 3 — Dual Strategy (Research + Commercial Split)
## Complete Strategic & Legal Analysis

> **Goal:** Use CC BY-NC-ND data for research/benchmarks AND build a
> commercially viable product with clean legal separation.

---

## 1. Legal Framework Analysis

### 1.1 CC BY-NC-ND 4.0 — Clause by Clause

**Section 2(a)(1) — License Grant:**
> "reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only"

**Meaning for us:**
- ✅ Can download, store, process NC data
- ✅ Can run experiments on NC data
- ✅ Can publish research papers using NC data results
- ❌ Cannot sell the data
- ❌ Cannot use data in a commercial product

**Section 2(a)(2) — NoDerivatives:**
> "No Derivatives — You may not Share Adapted Material"

**The critical question: Is a trained ML model "Adapted Material"?**

CC 4.0 defines "Adapted Material" as:
> "material that is derived from or based upon the Licensed Material"

**Three schools of thought:**

| View | Argument | Risk Level |
|------|----------|------------|
| **Conservative** | Model weights are mathematical transformations of training data → IS a derivative | 🔴 HIGH |
| **Moderate** | Model is a new creative work that was "inspired by" but doesn't contain the data → NOT a derivative | 🟡 MEDIUM |
| **Liberal** | Training = learning, not copying. Model learns patterns, not data → NOT a derivative | 🟢 LOW |

**Legal reality (as of 2026):**
- NO court has definitively ruled whether ML model weights are "derivative works"
- The EU AI Act and US copyright law are still evolving
- Most legal scholars lean toward the **moderate** view
- But for commercial safety, assume **conservative**

### 1.2 What Can We Legally Do with NC Data?

| Activity | Legal? | Basis |
|----------|--------|-------|
| Download NC data for internal R&D | ✅ Yes | Section 2(a)(1) — NonCommercial use allowed |
| Analyze NC data to understand patterns | ✅ Yes | Research use |
| Use NC data to design model architecture | ✅ Yes | Knowledge is not restricted |
| Use NC data to tune hyperparameters | ⚠️ Gray | Some argue this creates "derivative" knowledge |
| Train final commercial model on NC data | ❌ No | Direct violation of NC clause |
| Publish paper with NC data benchmarks | ✅ Yes | Academic use is NonCommercial |
| Show NC benchmarks in marketing | ⚠️ Gray | Marketing is commercial, but citing research is OK |
| Include NC data in GitHub repo | ⚠️ Gray | GitHub is public, some may consider it "sharing" |
| Sell model trained on NC data | ❌ No | Clear commercial violation |

### 1.3 Case Law & Precedents

**Key cases (as of 2026):**

| Case | Year | Relevance | Outcome |
|------|------|-----------|---------|
| **Andersen v. Stability AI** | 2023-ongoing | AI art training on copyrighted images | Pending — could set precedent |
| **NYT v. OpenAI** | 2023-ongoing | LLM training on copyrighted text | Pending |
| **Thomson Reuters v. ROSS** | 2024 | Legal data used for AI training | Settled — no precedent |
| **GitHub Copilot case** | 2022-ongoing | Code generation from open-source | Pending |

**No precedent specifically about CC BY-NC-ND + ML training yet.**

### 1.4 How Major Companies Handle This

| Company | Strategy | Details |
|---------|----------|---------|
| **Meta (LLaMA)** | Separate licenses | Research model (CC BY-NC) vs commercial model (different license) |
| **Mistral** | Clean data | Train only on permissive/open data |
| **Google (Gemini)** | Proprietary | Don't disclose training data sources |
| **Stability AI** | Mixed | LAION dataset (CC BY) for Stable Diffusion |
| **Hugging Face** | Dataset cards | Explicit license tracking per dataset |

**Key insight:** Companies that care about commercial use either:
1. Use only permissive data (Mistral approach) — safest
2. Maintain strict separation (Meta approach) — workable but risky
3. Don't disclose (Google approach) — risky if discovered

---

## 2. Technical Implementation Plan

### 2.1 Git Branching Strategy

```
main (commercial)
├── commercial/data/          ← Only permissive-licensed data
├── commercial/models/        ← Models trained on permissive data
├── commercial/training/      ← Training pipeline (clean)
│
├── research/data/            ← ALL data including NC (gitignored for main)
├── research/models/          ← Research models (NOT distributed)
├── research/experiments/     ← Notebooks, analysis
│
├── src/energivanu/           ← Shared code (data-agnostic)
├── config/                   ← Shared config
└── docs/                     ← Documentation (no NC references in main)
```

**Branch strategy:**
```bash
main                 # Commercial — NEVER contains NC data
├── research         # Research — can reference NC data
├── feature/xxx      # Feature branches — merge to main only if clean
└── experiment/xxx   # Experiment branches — stay in research
```

### 2.2 CI/CD Pipeline — Compliance Enforcement

```yaml
# .github/workflows/compliance-check.yml
name: License Compliance Check

on:
  pull_request:
    branches: [main]

jobs:
  check-nc-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check for NC-licensed data
        run: |
          # Scan all data files for NC dataset signatures
          NC_PATTERNS=(
            "FigShare"
            "MIT-Supercloud"
            "York University"
            "CC-BY-NC"
            "by-nc-nd"
          )

          for pattern in "${NC_PATTERNS[@]}"; do
            if grep -r "$pattern" data/ 2>/dev/null; then
              echo "❌ BLOCKED: NC-licensed data reference found: $pattern"
              exit 1
            fi
          done

          echo "✅ No NC-licensed data found in commercial branch"

      - name: Verify data license headers
        run: |
          # Check all CSV files have license header
          for f in data/**/*.csv; do
            head -1 "$f" | grep -q "license" || \
            head -1 "$f" | grep -q "CC-BY" || \
            echo "⚠️ Warning: $f missing license header"
          done

      - name: Check model training logs
        run: |
          # Verify training logs don't reference NC data
          if [ -f training.log ]; then
            if grep -i "york\|mit_supercloud\|figshare" training.log; then
              echo "❌ BLOCKED: Training log references NC data"
              exit 1
            fi
          fi
```

### 2.3 Documentation Strategy

**What to disclose (transparency):**
- ✅ "We use Alibaba Cluster Trace GPU v2020 (CC BY 4.0) for training"
- ✅ "We collect our own GPU telemetry data"
- ✅ "Our research uses various datasets for analysis (see research/ branch)"
- ✅ Cite all data sources in README and papers

**What NOT to include in commercial docs:**
- ❌ Don't mention York/MIT NC data in marketing
- ❌ Don't claim benchmarks from NC-trained models as commercial
- ❌ Don't include NC data references in main branch README
- ❌ Don't link to NC datasets from commercial pages

**Template for README (main branch):**
```markdown
## Training Data

This model was trained on:
1. **Alibaba Cluster Trace GPU v2020** (CC BY 4.0)
   - 6,500 GPUs, production workloads
   - Citation: Weng et al., NSDI '22

2. **Custom GPU Telemetry Data** (collected by us)
   - Multi-GPU, multi-workload time series
   - See `data/README.md` for details

All training data is commercially permissive.
```

### 2.4 Knowledge Transfer Problem

**The concern:** Can insights from NC data "contaminate" commercial work?

**Legal analysis:**

| Knowledge Type | Transferable? | Why |
|----------------|--------------|-----|
| "TCN works well for power prediction" | ✅ Yes | General knowledge, not data-specific |
| "15 features is the right number" | ✅ Yes | Architecture decision |
| "seq_len=30 works best" | ✅ Yes | Hyperparameter knowledge |
| "This specific weight value is optimal" | ❌ No | Derived from NC data |
| "MAPE of 1.85% is achievable" | ⚠️ Gray | Benchmark from NC data |

**Practical approach:**
1. Use NC data for: exploration, understanding, hypothesis generation
2. Validate all insights on permissive data before commercial use
3. If a finding from NC data doesn't hold on permissive data, discard it
4. Document the validation chain

---

## 3. Risk Assessment Matrix

| Risk | Probability | Impact | Severity | Mitigation |
|------|------------|--------|----------|------------|
| NC data accidentally in commercial training | Low | Critical | 🔴 HIGH | CI/CD compliance checks |
| Model weights ruled as "derivative work" | Medium | Critical | 🔴 HIGH | Train final model on permissive data only |
| Audit reveals NC data usage | Low | High | 🟡 MEDIUM | Clean git history, documentation |
| Contributor adds NC data to main | Medium | High | 🟡 MEDIUM | Branch protection, code review |
| Marketing accidentally cites NC benchmarks | Low | Medium | 🟡 MEDIUM | Separate research/commercial docs |
| Legal precedent unfavorable to ML | Low | Critical | 🟡 MEDIUM | Diversify data sources |
| Attribution requirement missed | Low | Low | 🟢 LOW | Checklist, automated checks |

---

## 4. Compliance Checklist

### Before Every Commercial Release:

- [ ] **Data audit:** All training data traced to permissive sources
- [ ] **No NC data:** Verified no CC BY-NC-ND data in training pipeline
- [ ] **Attribution:** All data sources cited correctly
- [ ] **Training logs:** Clean, no references to NC datasets
- [ ] **Model card:** Documents data sources, licenses
- [ ] **Git history:** No NC data ever committed to main branch
- [ ] **CI/CD passed:** Automated compliance checks green

### For Research Publications:

- [ ] **NC data OK:** Can use for research papers
- [ ] **Clearly labeled:** "Research model, not for commercial use"
- [ ] **Separate branch:** Research artifacts in research/ branch
- [ ] **Attribution:** All data sources cited

### For Marketing/Advertising:

- [ ] **Only commercial benchmarks:** Don't cite NC-trained model results
- [ ] **No NC dataset names:** Don't mention York/MIT in marketing
- [ ] **Legal review:** Have someone review marketing claims

---

## 5. Implementation Timeline

### Phase 1: Setup (Week 1)
| Day | Task | Owner |
|-----|------|-------|
| 1 | Create `research/` branch structure | Dev |
| 1 | Set up branch protection rules | DevOps |
| 2 | Create CI/CD compliance workflow | DevOps |
| 2 | Move existing NC data references to research/ | Dev |
| 3 | Create data license tracking document | Legal/PM |
| 3 | Set up data provenance logging | Dev |

### Phase 2: Data Collection (Weeks 2-3)
| Day | Task | Owner |
|-----|------|-------|
| 8 | Download Alibaba GPU Trace (CC BY 4.0) | Dev |
| 9 | Process Alibaba data into our format | Dev |
| 10 | Start own data collection (Option 1) | Dev |
| 12 | Combine Alibaba + own data | Dev |
| 14 | Validate combined dataset | Dev |

### Phase 3: Commercial Model Training (Week 4)
| Day | Task | Owner |
|-----|------|-------|
| 15 | Train model on permissive data only | Dev |
| 16 | Validate MAPE on held-out set | Dev |
| 17 | Export to ONNX | Dev |
| 18 | Create model card with full provenance | Dev |
| 19 | Run compliance checklist | Legal/PM |

### Phase 4: Research Parallel Track (Ongoing)
| Task | Owner |
|------|-------|
| Run experiments on NC data (research branch) | Research |
| Publish paper with NC data benchmarks | Research |
| Explore architecture improvements | Research |
| Document insights for commercial pipeline | Research |

---

## 6. Cost-Benefit Analysis

| Approach | Legal Risk | Cost | Time | Quality |
|----------|-----------|------|------|---------|
| NC data only (illegal) | 🔴 Critical | $0 | Fast | Best data |
| Permissive data only (Option 2) | 🟢 None | $0-30 | Medium | Good |
| Dual strategy (this) | 🟡 Low | $0-30 | Medium | Best of both |
| Own data only (Option 1) | 🟢 None | $0-30 | Slow | Variable |

**Dual strategy gives:**
- Research freedom (use any data for learning)
- Commercial safety (train final model on clean data)
- Best practices (industry-standard approach)
- Flexibility (can pivot if laws change)

---

## 7. Decision Framework

```
START
  │
  ├─ Need research benchmarks?
  │   ├─ YES → Use NC data in research/ branch ✅
  │   └─ NO → Skip NC data entirely
  │
  ├─ Training commercial model?
  │   ├─ YES → Use ONLY permissive data (Alibaba + own) ✅
  │   └─ NO → Any data OK for experiments
  │
  ├─ Publishing paper?
  │   ├─ YES → Can cite NC benchmarks (with attribution) ✅
  │   └─ NO → No citation needed
  │
  ├─ Marketing material?
  │   ├─ YES → Only commercial model benchmarks ✅
  │   └─ NO → No restrictions
  │
  └─ Open-sourcing on GitHub?
      ├─ YES → main branch = permissive data only ✅
      └─ NO → Any data OK internally
```

---

## 8. Verdict

| Criteria | Rating |
|----------|--------|
| Legal safety | ⭐⭐⭐⭐ (Safe if implemented correctly) |
| Research freedom | ⭐⭐⭐⭐⭐ (Full access to all data) |
| Commercial readiness | ⭐⭐⭐⭐ (Clean pipeline) |
| Effort | ⭐⭐⭐ (Medium — needs discipline) |
| Risk of contamination | ⭐⭐⭐ (Mitigated by CI/CD) |
| Industry alignment | ⭐⭐⭐⭐⭐ (Standard practice) |

**RECOMMENDATION: This is the BEST strategy if you want to maximize both research capability and commercial safety. Combine with Option 1 (own data) and Option 2 (Alibaba data) for the strongest position.**

---

## 9. Final Recommendation — Combined Approach

```
┌─────────────────────────────────────────────────┐
│           RECOMMENDED: ALL THREE OPTIONS         │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. OPTION 1 (Own Data)     → Primary source    │
│     - Kaggle/Colab collection                    │
│     - Full commercial rights                     │
│                                                  │
│  2. OPTION 2 (Alibaba CC BY) → Supplement       │
│     - 6500 GPUs, production data                 │
│     - CC BY 4.0, fully commercial                │
│                                                  │
│  3. OPTION 3 (Dual Strategy) → Framework        │
│     - Research/commercial separation             │
│     - CI/CD compliance enforcement               │
│     - Documentation strategy                     │
│                                                  │
│  RESULT: Bulletproof commercial model            │
│  with full research capability                   │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Priority order:**
1. ⚡ Start Option 1 NOW (Kaggle notebook, free, immediate)
2. 📥 Download Alibaba data (Option 2, free, this week)
3. 🔧 Set up dual strategy infrastructure (Option 3, week 1)
4. 🏋️ Train commercial model on clean data (week 4)
5. 📄 Publish research using NC data (ongoing)
