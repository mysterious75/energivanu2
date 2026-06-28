# Legal FAQ — Energivanu Licensing & Data Compliance

Frequently asked questions about using Energivanu commercially, data licensing, and legal considerations.

---

## Table of Contents

1. [Can I use Energivanu commercially?](#1-can-i-use-energivanu-commercially)
2. [What about the York University data?](#2-what-about-the-york-university-data)
3. [Which licenses are safe for commercial use?](#3-which-licenses-are-safe-for-commercial-use)
4. [How do I cite the Alibaba data?](#4-how-do-i-cite-the-alibaba-data)
5. [Are model weights a "derivative work"?](#5-are-model-weights-a-derivative-work)
6. [What if I train on my own data?](#6-what-if-i-train-on-my-own-data)
7. [Can I use this in a SaaS product?](#7-can-i-use-this-in-a-saas-product)
8. [Liability disclaimer](#8-liability-disclaimer)

---

## 1. Can I use Energivanu commercially?

**Yes.** The Energivanu code is licensed under **AGPL-3.0-or-later**, which permits commercial use.

However, there are two considerations:

- **Code license (AGPL-3.0)**: You can use, modify, and distribute the code commercially. If you run a modified version as a network service (SaaS), you must release your modified source code under AGPL-3.0.
- **Model weights**: The distributed demo model (`best_model_demo.pt`) is trained on synthetic data — fully safe for commercial use. The commercial model (`commercial_best.pt`) is trained on Alibaba CC BY 4.0 + own data — also fully safe.

**If you need a proprietary license** (no AGPL obligations), contact us for a separate commercial license.

---

## 2. What about the York University data?

The York University H100 dataset is licensed under **CC BY-NC-ND 4.0**:

- **NC (Non-Commercial)**: You cannot use the data for commercial purposes.
- **ND (No Derivatives)**: You cannot distribute derivative works (potentially including trained model weights).

### What this means for you:

| Activity | Allowed? |
|----------|----------|
| Use York data for personal research | ✅ Yes |
| Use York data to learn about GPU power patterns | ✅ Yes |
| Include York data in commercial training pipeline | ❌ No |
| Distribute model weights trained on York data | ❌ No |
| Use York data for hyperparameter tuning (research) | ⚠️ Gray area (we don't) |

### Our approach:

We **never** include York data in any commercial training pipeline. The `train_commercial.py` script explicitly blocks non-commercial sources. All distributed model weights are trained only on:

1. Alibaba GPU Trace (CC BY 4.0) — commercial safe
2. Self-collected data (we own it) — commercial safe
3. Synthetic data (we generated it) — commercial safe

---

## 3. Which licenses are safe for commercial use?

| License | Commercial Use | Can Train Models | Can Distribute Weights | Notes |
|---------|---------------|-----------------|----------------------|-------|
| **CC BY 4.0** | ✅ Yes | ✅ Yes | ✅ Yes | Must cite source |
| **Apache 2.0** | ✅ Yes | ✅ Yes | ✅ Yes | Patent grant included |
| **MIT** | ✅ Yes | ✅ Yes | ✅ Yes | Minimal restrictions |
| **CC0 / Public Domain** | ✅ Yes | ✅ Yes | ✅ Yes | No restrictions |
| **CC BY-NC 4.0** | ❌ No | ⚠️ Research only | ❌ No | NC = non-commercial |
| **CC BY-NC-ND 4.0** | ❌ No | ⚠️ Research only | ❌ No | Most restrictive |
| **CC BY-SA 4.0** | ✅ Yes | ✅ Yes | ⚠️ Must share alike | Copyleft for data |

### Safe sources we use:

| Source | License | Status |
|--------|---------|--------|
| Alibaba GPU Trace v2020 | CC BY 4.0 | ✅ Commercial safe |
| Self-collected T4 data | Own | ✅ No restrictions |
| Synthetic data | N/A | ✅ No restrictions |

### Blocked sources:

| Source | License | Status |
|--------|---------|--------|
| York University H100 | CC BY-NC-ND 4.0 | ❌ Research only |
| MIT Supercloud | CC BY-NC-ND 4.0 | ❌ Research only |

---

## 4. How do I cite the Alibaba data?

The Alibaba GPU Trace v2020 is licensed under CC BY 4.0, which requires attribution.

### In a paper:

```bibtex
@inproceedings{weng2022mlaas,
  title={{MLaaS} in the Wild: Workload Analysis and Scheduling
         in Large-Scale Heterogeneous {GPU} Clusters},
  author={Weng, Qizhen and Xiao, Wencong and Yu, Yinghao and
          Wang, Wei and Wang, Cheng and He, Jian and Li, Yong
          and Zhang, Liping and Lin, Wei and Ding, Yu},
  booktitle={NSDI '22},
  year={2022}
}
```

### In a README or documentation:

```
This product uses data from the Alibaba Cluster Trace GPU v2020 dataset,
licensed under CC BY 4.0. Source: https://github.com/alibaba/clusterdata
```

### In a commercial product:

Include the citation in your documentation, about page, or credits section. The CC BY 4.0 license does not require attribution in advertising, but it's good practice.

---

## 5. Are model weights a "derivative work"?

This is the most important legal question in ML licensing, and **there is no definitive court ruling as of 2026**.

### Two schools of thought:

**Conservative view (what we follow):**
- Model weights trained on data are a "derivative work" of that data
- If the data is CC BY-NC-ND, the weights cannot be distributed commercially
- This is the safest legal position

**Liberal view:**
- Model weights are a "new work" inspired by data, not a copy
- The data is an input, like a textbook is an input to a student
- Training is "fair use" or equivalent

### Our position:

We take the conservative approach:
- **Never** distribute weights trained on NC-licensed data
- **Always** use commercially-licensed data for distributable models
- **Document** exactly which data sources were used for each checkpoint

### What this means for you:

If you train Energivanu on your own data:
- **Your data, your weights** — no restrictions
- If you use NC data, don't distribute the weights commercially
- If you use CC BY data, cite the source

---

## 6. What if I train on my own data?

**You own the weights.** If you collect GPU telemetry from your own hardware:

- ✅ You can use the weights however you want
- ✅ You can distribute them commercially
- ✅ You can sell them
- ✅ No attribution required (it's your data)

This is the recommended approach for production deployments. Use `train_commercial.py` with your own collected data:

```bash
python -m energivanu.train_commercial --sources kaggle_t4
```

---

## 7. Can I use this in a SaaS product?

**Yes, but with AGPL obligations.**

The AGPL-3.0 license requires:
- If you modify Energivanu and run it as a network service, you must make your modified source code available under AGPL-3.0
- Users interacting with your service over the network must be able to download the source

### Options:

| Scenario | License | Requirement |
|----------|---------|-------------|
| Use unmodified Energivanu as API | AGPL-3.0 | Share source of any modifications |
| Modify Energivanu, run as SaaS | AGPL-3.0 | Must release modified source |
| Embed in proprietary product | Need commercial license | Contact us |

### If you don't want AGPL obligations:

Contact us for a separate commercial license that allows proprietary use without source disclosure requirements.

---

## 8. Liability Disclaimer

**Energivanu is provided "as is" without warranty of any kind.**

### Key disclaimers:

1. **No warranty of accuracy**: Power predictions are estimates. Do not rely on them as the sole input for safety-critical decisions.

2. **No warranty of fitness**: The software may not be suitable for your specific use case, hardware configuration, or regulatory environment.

3. **No liability for damages**: The authors and contributors are not liable for any damages arising from use of this software, including but not limited to:
   - Incorrect power predictions leading to grid instability
   - Battery damage from incorrect BESS dispatch signals
   - Financial losses from inaccurate demand forecasting
   - Regulatory violations from improper data handling

4. **User responsibility**: You are responsible for:
   - Validating the model on your specific hardware and workload
   - Ensuring compliance with your local regulations
   - Implementing appropriate safety margins
   - Not using the model as the sole decision-maker for critical systems

5. **Data compliance**: You are responsible for ensuring that any data you use with Energivanu complies with its license terms. We provide tools and documentation to help, but ultimate compliance is your responsibility.

### In plain English:

This is open-source research software. It's useful for learning and experimentation. For production use, you need to validate it thoroughly on your own systems. We're not responsible if something goes wrong.

---

## Still have questions?

- **GitHub Issues**: [mysterious75/Energivanu](https://github.com/mysterious75/Energivanu/issues)
- **Twitter/X**: [@VEDKUMAR98](https://x.com/VEDKUMAR98)
- **Commercial licensing**: Contact via GitHub Issue or Twitter
