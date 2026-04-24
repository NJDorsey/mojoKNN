# Empirical Runtime Complexity Analysis: k-d Tree Extension

## 1. Project Context
This research project extends the work of **Kolli et al.**, which evaluated the performance of Mojo-based $k$-NN implementations. The original paper used empirical runtime complexity to compare Python and Mojo across various hardware environments. 

This extension introduces a **k-d tree** spatial data structure to replace the original brute-force search. While the original study optimized the language-level overhead ($C$), this study optimizes the algorithmic scaling exponent ($k$).

---

## 2. Mathematical Model
We adopt the power-law model used in the base paper to describe runtime $T$ as a function of dataset size $n$:

$$T(n) \approx C \cdot n^k$$

### Components:
* **$k$ (Complexity Exponent):** Reflects the scaling behavior. A value of $k \approx 1$ suggests near-linear scaling, while $k \approx 2$ suggests quadratic scaling.
* **$C$ (Constant Factor):** Represents the baseline runtime cost independent of $n$. This captures low-level efficiency such as memory layout, cache utilization, and language overhead.

### The Log-Log Regression:
To estimate these parameters, we transform the power-law equation into a linear form using logarithms:

$$\log(T(n)) = k \log(n) + \log(C)$$

By plotting $\log(n)$ on the x-axis and $\log(T)$ on the y-axis, we can perform a linear regression where:
1.  The **Slope** of the line is $k$.
2.  The **Y-intercept** of the line is $\log(C)$.

---

## 3. Implementation Specifics (k-d Tree)
Because a $k$-d tree involves a "build" phase and a "search" phase, we must perform two separate regressions to fully characterize the performance:

| Measurement | Scope | Expected Result |
| :--- | :--- | :--- |
| **Total Time** | Tree Construction + Query | Higher $C$ (setup cost), Lower $k$ (better scaling). |
| **Query-Only Time** | Search on pre-built tree | Lowest $C$, $k$ should approach theoretical $\log(n)$ behavior. |

---

## 4. Experimental Requirements
To generate a reliable regression on a high-end PC, the following protocol must be observed:

### Data Selection
* **Dataset:** AAPL stock dataset (approx. 350,000 rows).
* **Sampling:** Use **logarithmic spacing** for $n$ to ensure even distribution on the log-log plot (e.g., $n = [1000, 3000, 10000, 30000, 100000, 350000]$).

### Benchmark Controls
1.  **Warm-up Phase:** Run the algorithm 3–5 times before recording to ensure the CPU is at peak clock frequency and the cache is "warm."
2.  **Median Averaging:** For each $n$, run 5 trials and record the **median** time to filter out background OS noise.
3.  **High-End System Context:** Be prepared for a "steeper" $k$ at very large $n$ values, as even a $k$-d tree will eventually hit the memory bandwidth ceiling (the "Memory Wall").

---

## 5. Objectives
1.  **Quantify the Scaling Improvement:** Compare the new $k$ (target $< 1.2$) against the original $k \approx 1.33$.
2.  **Identify the Crossover Point:** Determine the dataset size $n$ where the $k$-d tree becomes faster than the brute-force method.
3.  **Validate Mojo Efficiency:** Confirm if $C$ remains low on high-end hardware despite the added complexity of tree management.