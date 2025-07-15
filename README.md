# 🩺 Medical-Order Extraction from Doctor–Patient Dialogues  
*Solo-Task submission · 2ᵈ place on the public leaderboard*

> “Reduce documentation burden, surface critical orders, and let clinicians focus on care.”

---

## TL;DR

This notebook demonstrates an end‑to‑end pipeline for **medical‑order extraction** from doctor–patient conversations. It ingests the *ACI‑Bench* and *PriMock57* transcripts, applies advanced prompt‑engineering around **Gemini 2.5 Pro** (You can use it on any closed source models like **Mistral Medium 3.2**, etc), post‑processes and normalises the JSON output, and writes leaderboard‑ready predictions. Earlier iterations using BioBERT, DeBERTa‑v3, BioClinical‑ModernBERT, T5, MedGemma 4B and Lingshu 7B informed the final design but were ultimately outperformed by a pure‑LLM + chain‑of‑thought + Self-Critique + Zero Shot strategy. All key steps—from data loading to cleaning—are modular, reproducible, and GPU‑free.

---

## 1  |  Project Background

### 1.1 Task

Extract structured orders ( `order_type`, `description`, `reason`, `provenance` ) from lengthy conversational transcripts to lighten clinical documentation burden and preserve critical patient information. The shared task ranks submissions by **ROUGE‑1 F1** on descriptions and reasons, plus strict F1 on order‑type and multi‑label provenance.([ACL Anthology][1])
Leaderboard scoring averages four metrics: `description_Rouge1_f1`, `reason_Rouge1_f1`, `order_type_Strict_f1`, and `provenance_MultiLabel_f1`.

### 1.2 Datasets

| Split  | Encounters | Orders | Sources                   |
| ------ | ---------: | -----: | ------------------------- |
| Train  |         63 |    143 | *ACI‑Bench* + *PriMock57* |
| Dev    |        100 |    255 | same                      |

* **ACI‑Bench** captures free‑flowing clinic visits without virtual‑assistant cues.([PubMed Central][2])
* **PriMock57** contains 57 mock primary‑care consultations with gold transcripts and notes.([GitHub][3], [ACL Anthology][4])

---

## 2  |  Model‑Selection Journey

| Phase | Goal                    | Model(s)                                                                                           | Rationale                                | Outcome                                            |
| ----- | ----------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------- |
|  ①    |  Classify order‑type    | **BioBERT**([arXiv][5])                                                                            | domain‑specific encoder                  | weak recall                                        |
|  …    | NER for attributes      | **DeBERTa‑v3**([Hugging Face][6])                                                                  | superior token‑level F1                  | inconsistent spans                                 |
|  ②    | Joint class‑&‑generate  | **BioClinical‑ModernBERT**([arXiv][7], [Hugging Face][8])  +  **T5**([arXiv][9])                   | long‑context encoder + versatile decoder | better order separation but low narrative fidelity  |
|  ③    | End‑to‑end generation   | **MedGemma 4B**([Google for Developers][10]), **Lingshu 7B**([arXiv][11])                            | SOTA medical LLMs                        | GPU OOM on Colab even with 4-bit quant                             |
|  ④    | Prompt‑engineering only | **Gemini 2.5 Pro**([Google DeepMind][12]), **Mistral Medium 3.2**([Mistral AI Documentation][13]) | zero‑server costs, strong reasoning      | **🥈 0.601 avg score with zero training** (≈ +13 F1 over phase ②)        |

> **Key insight:** a carefully crafted *CoT → self‑verify → final‑JSON* prompt with a deterministic seed delivered more gains than custom fine‑tuning under tight compute budgets.

---

## 3  |  Notebook Overview

| Cell                 | Purpose                                               |
| -------------------- | ----------------------------------------------------- |
|  1  Environment      | Mount Drive, set paths, install `google-generativeai` |
|  2  System prompt    | Immutable state‑machine instruction                   |
|  3  `generate()`     | Typed, logging‑aware wrapper around Gemini API        |
|  4  Data load        | Read `test_input_data.json`                           |
|  5  Batch run        | Stream results with progress bar & robust parsing     |
|  6  Retry logic      | One‑pass recovery of failed IDs                       |
|  7  Persist raw      | `test_output_data_raw.json`                           |
|  8  Utils            | Extract last valid JSON block, remove CoT, parse JSON |
|  9  Normalise & save | Flatten keys, produce `test_output_data_clean.json`   |

---

## 4  |  Getting Started

```bash
# clone and open in Colab
!git clone https://github.com/your‑org/med‑order‑extractor
```

1. **Set the `GOOGLE_API_KEY`** in Colab’s *User Settings ▸ Secrets*.
2. Run the notebook top‑to‑bottom (GPU **not** required).
3. Submit the zipped `test_output_data_clean.json` as `pred_orders.json`.

---

## 5  |  LeaderBoard Results

| Rank | description<br>Rouge-F1 | reason<br>Rouge-F1 | order_type<br>Strict-F1 | provenance<br>ML-F1 | **Avg** |
|------|------------------------:|-------------------:|------------------------:|--------------------:|---------|
| #1 (SOTA) | **0.6677** | 0.2949 | **0.8145** | **0.6304** | **0.6019** |
| **#2 (Ours)** | 0.6406 | **0.4130** | 0.7474 | 0.6044 | **0.6014** |
| #3 | 0.5799 | 0.3564 | 0.7156 | 0.4838 | 0.5339 |

---

## 6  |  Cite & Acknowledge

* Lee et al., “BioBERT: a pre‑trained biomedical language model” (2019)([arXiv][5])
* He et al., “DeBERTa v3: Improving DeBERTa using ELECTRA‑style pre‑training” (2021)([Hugging Face][6])
* Sounack et al., “BioClinical ModernBERT” (2025)([arXiv][7])
* Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text‑to‑Text Transformer” (T5)([arXiv][9])
* Google Health AI, “MedGemma 4 B Model Card” (2025)([Google for Developers][10])
* Xiao et al., “Lingshu: A Generalist Foundation Model for Unified Multimodal Medicine” (2025)([arXiv][11])
* DeepMind, “Gemini 2.5 Pro” (2025)([Google DeepMind][12])
* Mistral AI, “Models Overview” (2025)([Mistral AI Documentation][13])
* Korfiatis et al., “PriMock57 Dataset” (2022)([ACL Anthology][4])
* Ouyang et al., “ACI‑Bench: Ambient Clinical Intelligence Dataset” (2023)([PubMed Central][2])

---

### Happy extracting! 🎯

[1]: https://aclanthology.org/2021.bionlp-1.8/ "Overview of the MEDIQA 2021 Shared Task on Summarization in ..."
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10482860/ "Aci-bench: a Novel Ambient Clinical Intelligence Dataset for ..."
[3]: https://github.com/babylonhealth/primock57 "babylonhealth/primock57: Dataset of 57 mock medical ... - GitHub"
[4]: https://aclanthology.org/2022.acl-short.65/ "PriMock57: A Dataset Of Primary Care Mock Consultations"
[5]: https://arxiv.org/abs/1901.08746 "BioBERT: a pre-trained biomedical language representation model for biomedical text mining"
[6]: https://huggingface.co/microsoft/deberta-v3-base "microsoft/deberta-v3-base - Hugging Face"
[7]: https://arxiv.org/abs/2506.10896 "BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP"
[8]: https://huggingface.co/collections/thomas-sounack/bioclinical-modernbert-681b824d12b9b6899841f8c7 "BioClinical ModernBERT - a thomas-sounack Collection"
[9]: https://arxiv.org/abs/1910.10683 "Exploring the Limits of Transfer Learning with a Unified Text-to-Text ..."
[10]: https://developers.google.com/health-ai-developer-foundations/medgemma/model-card "MedGemma model card | Health AI Developer Foundations"
[11]: https://arxiv.org/html/2506.07044v1 "Lingshu: A Generalist Foundation Model for Unified Multimodal ..."
[12]: https://deepmind.google/models/gemini/pro/ "Gemini 2.5 Pro - Google DeepMind"
[13]: https://docs.mistral.ai/getting-started/models/models_overview/ "Models Overview - Mistral AI Documentation"
