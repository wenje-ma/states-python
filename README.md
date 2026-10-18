# Ancient Glass Composition Analysis (CUMCM 2022, Problem C)

> Complete entry for the 2022 CUMCM (China Undergraduate Mathematical Contest in Modeling) Problem C — *"Composition analysis and identification of ancient glass artifacts"* — including the problem statement, data, Python code, LaTeX paper, and slides.

> **Note on the repository name:** `states-python` refers to the weathering-*state* analysis in the paper, not U.S. states data.

## Problem

Given chemical-composition measurements of ancient glass artifacts (some weathered, some not; high-potassium vs. lead–barium types), the task was to: analyze the relationships between weathering, type and composition; divide the glass into sensible subcategories; predict the category of 8 unknown artifacts; and assess sensitivity to sampling perturbation.

## Repository structure

```
states-python/
├── 1-题目/                    # Problem statement (C题.md) + 3 attachment CSVs
├── 2-代码/                    # Working code, organized by question
│   ├── 0-数据预处理/          # Preprocessing.py + cleaned/ (20+ cleaned CSVs) + 伪代码.md/.pdf
│   ├── 1-第一问/              # Q1 script + result xlsx
│   ├── 2-第二问/              # Q2 script + result xlsx
│   └── 3-第三问/              # Q3 script + result xlsx
├── 2-解答/                    # Submission mirror of 2-代码/ (results only)
├── 3-报告/  3-论文/           # 论文.md / 论文.tex / 论文.pdf, 幻灯片.pdf, ref.bib, figures/
├── 3赵宇翔李传浩马文杰.zip     # Original submission archive
└── 4-文献/                    # 8 reference papers (PDF+TXT) + 9 course handouts (MD)
```

## Method pipeline (from the paper)

《基于成分数据建模与多模型融合的古代玻璃制品成分分析与鉴别》 — *"Composition analysis and identification of ancient glass artifacts based on compositional data modeling and multi-model fusion"*:

1. **Filtering** — keep samples whose component sums fall in 85%–105%;
2. **SVR imputation** for missing composition values;
3. **ALR/CLR log-ratio transforms** to handle compositional-data constraints;
4. **Q1** — χ² tests / Monte-Carlo permutation + Spearman/Kendall show weathering is significantly associated with type; Mann–Whitney U + Cohen's d quantify composition differences;
5. **Q2** — random-forest feature selection + a fusion of logistic regression / random forest / depth-limited decision trees, with **K-means++ subclustering** (K=4 for high-potassium, K=6 for lead–barium);
6. **Q3** — classify the 8 unknown artifacts (A1/A6/A7 → high-potassium; the rest → lead–barium) with perturbation sensitivity analysis.

## Language

Problem statements, paper, references and code comments are in **Chinese**. The paper exists as both Markdown (`论文.md`) and LaTeX (`论文.tex` + `ref.bib`) with a compiled `论文.pdf` and slide deck (`幻灯片.pdf`).

## Reproducing

Run the Python scripts in `2-代码/` in order (0-数据预处理 → 第一问 → 第二问 → 第三问); each question script writes its result `xlsx`. No special environment file is provided — standard scientific Python (pandas, scikit-learn) suffices.
