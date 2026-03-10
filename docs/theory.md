# 📑 Mathematical Foundation: Shapley Values

The core of this toolbox is based on **Shapley Additive Explanations (SHAP)**, which applies a game-theoretic approach to model transparency. The goal is to assign each feature a value that represents its contribution to the change in the model output from the baseline (average) value.

### The Shapley Equation
The contribution of a feature $i$ is defined by the following formula:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n - |S| - 1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

**Where:**
* **$n$**: The total number of input features.
* **$S$**: A subset (coalition) of features that does **not** include feature $i$.
* **$v(S)$**: The model's prediction using only the features in subset $S$.
* **$v(S \cup \{i\}) - v(S)$**: The **marginal contribution**—how much the prediction changes when feature $i$ is added to the coalition.
* **$\frac{|S|!(n - |S| - 1)!}{n!}$**: The **weighting factor**, representing the probability of the coalition $S$ occurring across all possible permutations.

---

### How are they computed?
Calculating exact Shapley values is computationally expensive because it requires evaluating $2^n$ possible combinations. This toolbox utilizes efficient approximation methods (like **TreeSHAP** or **KernelSHAP**) that follow a "leave-one-out" logic:

1.  **Coalition Sampling:** The algorithm samples various combinations (coalitions) of input features.
2.  **Marginal Contribution:** For each sampled coalition, it measures the difference in the model’s output with and without a specific feature.
3.  **Weighted Averaging:** It aggregates these differences, weighting them to ensure that the order in which features are added does not bias the final value.



---

### The Four Essential Axioms
The power of SHAP lies in four mathematical guarantees that ensure the explanations are "fair" and consistent:

* **Efficiency (Additivity):** The sum of all feature contributions ($\phi_i$) plus the expected base value exactly equals the model output. This ensures the explanation is "honest" to the prediction.
* **Symmetry:** If two features contribute equally to all possible coalitions, their Shapley values will be identical.
* **Dummy (Nullity):** A feature that has no impact on any coalition will have a Shapley value of exactly zero.
* **Linearity:** If a model is a linear combination of sub-models, the Shapley value of the total model is the sum of the individual Shapley values.

---

### Why this matters for the Energy Sector
In high-stakes energy applications—such as **reserve bidding** or **grid congestion management**—understanding the marginal contribution is critical. Unlike global importance scores, Shapley values provide **local explanations**. 

They allow operators to understand why *this specific hour* had a high price forecast or a solar drop, identifying whether it was driven by low wind speeds, a specific plant outage, or unexpected demand shifts. SHAP enables the deployment of AI in regulated, safety-critical environments.


