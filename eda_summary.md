## 🧠 Exploratory Data Analysis Summary – UCF-Crime Text Dataset

---

### **1. Dataset Overview**

* Parsed from `UCFCrime_Train.json`: **15,677 sentences** from **1,165 videos** across **14 crime categories** plus *Normal_Videos*.
* Each record contains *video ID, label, concatenated sentence description, duration, and sentence count*.
* Dataset is **highly imbalanced**, dominated by *Normal_Videos* (~590) versus minority classes (*Explosion, Arson, Assault*).

---

### **2. Sentence-Level and Video-Level Analysis**

* **Sentence length:** 10–25 words on average (short, descriptive captions).
* **Text length (per video):** heavy-tailed — most under 1000 words, some exceed 20 000 words.
* **Frequent words:** dominated by visual-action terms (“man”, “car”, “walked”, “door”).
* **After cleaning/lemmatization (spaCy):**

  * *Arson*: “gasoline”, “fire”, “burned”.
  * *Assault*: “beat”, “ground”, “stick”.
  * *Robbery*: “counter”, “gun”.
  * *Normal*: “screen”, “video”, “road”.

These lexical fields clearly capture visual context per crime.

---

### **3. Semantic Overlap and Embeddings**

* Word overlap between *Normal* and other classes exceeds 400–700 shared tokens per pair, confirming strong lexical redundancy.
* **SentenceTransformer (MiniLM)** embeddings + PCA show partial clustering — *Normal* forms a dense core, while crime classes scatter.
  → Raw sentence space offers limited discriminative structure.

---

### **4. Binary & Multi-Class Contrastive Fine-Tuning**

* Binary mapping (*Violent* vs *Non-Violent*) reduces to ≈330 vs 830 samples.
* Fine-tuned `all-mpnet-base-v2` using **CosineSimilarityLoss** for 1 epoch:

  * **Before tuning:** violent/non-violent overlap heavily.
  * **After tuning:** clear bimodal separation in PCA space.
* **Average cosine similarities:**

  * Violent intra-class = 0.185
  * Non-violent intra-class = 0.099
  * Cross-class = 0.012
    → Tuned embeddings cluster by semantic violence intensity.

---

### **5. Embedding Geometry & Multi-Class Patterns**

* **Global cosine similarity:** 0.071 ± 0.133.
* **Per-class intra-cosine (cohesion):**

  * *Arrest* (0.54) > *Assault* > *Arson* > *Fighting* > *Robbery*.
  * *Normal_Videos* are least cohesive (0.14).
* **Centroid distances (PCA):**

  * *Assault–Fighting*: 0.03 → close semantic pair.
  * *Burglary–Robbery*: 0.19 → similar.
  * *Normal–Explosion*: ~0.6 → farthest pair.
* **Embedding norms:** stable (≈ 1.0 ± 0.1), no collapse.
* **Multi-class cosine heatmap:** largely uncorrelated — good class diversity.

---

### **6. Weighted TF-IDF Analysis**

* Generated **TF-IDF weighted word clouds** highlighting discriminative lexicon:

  * *Abuse*: “child”, “dog”, “wheelchair”.
  * *Arrest*: “police”, “uniform”, “handcuffs”.
  * *Arson*: “gasoline”, “burning”, “house”.
  * *Assault*: “beat”, “stick”, “ground”.
  * *Normal*: “video”, “screen”, “car”.
    → Distinct thematic focus per class, confirming linguistic validity of labels.

---

### **7. Class Imbalance and Weighting**

* **Balanced weights (inverse frequency):**

| Class                   | Weight |
| :---------------------- | :----: |
| Explosion               |  11.1  |
| Arson                   |   4.3  |
| Assault                 |   3.1  |
| Shoplifting / Vandalism |   2.6  |
| Abuse / Fighting        |  ≈ 2.1 |
| Normal_Videos           |  0.17  |

* Bar chart confirms extreme imbalance (minority-to-majority ratio ≈ 65×).

---

### **8. Linear Probe on Fine-Tuned Embeddings**

* A **Logistic Regression probe** (weighted) achieves:

  * **Accuracy:** 0.74
  * **Macro F1:** 0.67
  * **Weighted F1:** 0.75
* **Per-class highlights:**

  * *RoadAccidents*: perfect (1.00 F1).
  * *Arrest, Fighting, Burglary*: >0.8 F1.
  * *Explosion* & *Assault*: underperform due to small sample counts.
* Confusion matrix shows most confusion among visually similar events (*Shoplifting ↔ Stealing*, *Normal_Videos ↔ Normal_Videos_*).

---

### **9. Hard-Negative Mining (Cross-Class Similarity)**

* Compared cosine similarities between *Violent* and *Non-Violent* embeddings:

  * Top pairs include:

    * *Explosion ↔ Arson*: 0.79
    * *Explosion ↔ RoadAccidents*: 0.69
    * *Burglary ↔ Stealing*: 0.63
  * These are **semantically hard negatives** — visually similar but distinct crime labels.
* Exported top pairs (`hard_negatives_pairs.csv`) for iterative fine-tuning.

---

### **✅ Overall Insights**

1. **Textual redundancy** makes naive classification difficult — embeddings are essential.
2. **Contrastive fine-tuning** significantly increases inter-class separability.
3. **Minor classes** (e.g., *Explosion*) remain bottlenecks; reweighting or augmentation required.
4. **Linear probing** confirms embeddings capture crime semantics (F1 ≈ 0.75 weighted).
5. **Hard-negative mining** identifies borderline cases for future supervised contrastive retraining.

Perfect — these figures complete the **“Hard Negative Contrastive Fine-Tuning”** section beautifully. Let’s finalize this by integrating the **new histograms, UMAP visualization, and separability metrics** into the master EDA report.

---

## 🧩 10. Hard-Negative Mining & Contrastive Re-Training

### **10.1 Distribution of Cross-Class Similarities**

After extracting the hardest Violent → Non-Violent pairs from the tuned embeddings, the cosine similarity histogram reveals:

* Mean similarity ≈ 0.43 (moderately high),
* Tail up to ≈ 0.8 — strong semantic overlap between certain crimes and benign actions.
* This distribution confirms that most “hard negatives” reside in a mid-similarity region, perfect for discriminative fine-tuning.

---

### **10.2 Class-wise Difficulty (Average Hard Negative Similarity)**

When grouping by *violent class*, the mean similarity to its nearest *non-violent* counterpart ranks as:

| Rank | Violent Label                           | Avg. Similarity | Interpretation                           |
| :--: | :-------------------------------------- | :-------------: | :--------------------------------------- |
|   1  | **Explosion**                           |       0.61      | visually confusable with road accidents  |
|   2  | **Assault**                             |       0.49      | frequent overlap with group interactions |
|   3  | **Arrest**                              |       0.48      | similar to controlled police scenes      |
|   4  | **Abuse**                               |       0.46      | overlaps with calm domestic actions      |
|  5–8 | *Fighting, Shooting, Burglary, Robbery* |   0.42 ± 0.01   | moderate difficulty                      |

Explosion and Assault emerge as the *hardest* classes to separate from Non-Violent segments, confirming that visual-text cues (e.g., “cars”, “fire”, “people walking”) appear across both domains.

---

### **10.3 Spatial Visualization of Hard Negatives**

A 2-D UMAP projection of the entire embedding space highlights the Violent (red) and Non-Violent (blue) regions, with grey connectors linking mined hard pairs.
These links concentrate along cluster borders — the semantic “decision boundary.”
Such visualization provides an interpretable geometric map of ambiguity within the dataset.

---

### **10.4 Post-Hard-Negative Contrastive Fine-Tuning**

A second-stage contrastive training (2 epochs, ≈ 1.8k pairs) incorporating these mined pairs yields a noticeably **cleaner separation** in PCA space:

* The Violent and Non-Violent clusters expand apart with reduced overlap.
* Qualitatively, cluster density and intra-class cohesion increase.

---

### **10.5 Quantitative Embedding Quality (Binary Separability)**

| Metric                | Pre-Hard-Neg | Post-Hard-Neg | Δ (Improvement ↓ = better) |
| :-------------------- | :----------: | :-----------: | :------------------------: |
| **Silhouette Score**  |     0.032    |   **0.047**   |           ↑ (15%)          |
| **Calinski–Harabasz** |     38.2     |    **55.1**   |           ↑ (44%)          |
| **Davies–Bouldin ↓**  |     4.38     |    **4.08**   |           ↓ (7%)           |

These metrics confirm an incremental but consistent improvement in **class compactness and separability**, validating the efficacy of the hard-negative step.

---

### **10.6 Key Insight**

Hard-negative mining acts as a **semantic sharpening mechanism**.
Rather than collapsing the representation space, it **stretches the decision boundary** to better discriminate between conceptually adjacent behaviors (e.g., “Explosion” ↔ “Road Accident”).
This mirrors techniques used in large-scale contrastive vision models (e.g., CLIP’s *hard-negative mining* for text-image alignment).

