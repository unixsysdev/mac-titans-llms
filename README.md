# Titans MAC (Memory as Context) - Investigation & Evolution

## Executive Summary
**Current Status**: Phase 1 (Passive Reconstruction) Complete.  
**Key Finding**: The **Addressing Gap**. The model successfully learns from streaming data via the surprise mechanism and injects that data into the residual stream, but **Identity Mapping ($f(x) \approx x$)** is insufficient for association. 

To solve "Needle in a Haystack" and "Rule Learning," we must shift from **Passive Echoing** to **Active Differentiable Search**.

---

## 1. Phase 1 Archive: Experimental Results (Passive MAC)

**TL;DR**: MAC produces measurable text variation but fails at fact retrieval and style transfer due to the lack of an associative indexing mechanism.

### Experimental Outcomes

| Experiment | Task | Status | Root Cause of Failure |
| :--- | :--- | :--- | :--- |
| **Exp 1** | Alien Language (Glorp) | ❌ **Fail** | Reconstruction $\neq$ Association. |
| **Exp 2** | Pirate Style Transfer | ❌ **Fail** | Passive residuals are too diluted to shift global style. |
| **Exp 3** | Pattern Completion | ❌ **Fail** | Requires active computation, not identity mirroring. |
| **Exp 4** | Text Reproduction | ❌ **Fail** | Factual recall requires a "Search Bar" (Query). |
| **Exp 6** | Injection Diagnostic | ✅ **SUCCESS** | **Proof of Life.** MAC changes the model's trajectory. |

### Phase 1 Diagnostic Conclusion
The implementation is technically sound—the "Surprise" triggers and injection occurs. The failure is **architectural**: The model stores information as a *numerical anomaly* in the weights, but tries to retrieve it as a *semantic concept* via the residual stream. These two spaces are currently unaligned.

---

## 2. Phase 2 Architecture: Active Differentiable Search

To bridge the retrieval gap, we are replacing the **Identity-based MLP** with a **Latent Query Loop**. Instead of the MAC "pushing" a residual, the Transformer "pulls" specific data via a searchable index.

### Core Components
* **The Index (MAC Weights):** Functions as a high-dimensional neural vector database.
* **The Query (Latent Intent):** A vector produced by the model representing the "search intent" (e.g., *"I need the info related to this high-surprise moment"*).
* **Differentiable Refinement:** The model uses the **gradient of search quality** to "nudge" its internal query until it resonates with the specific weights where the information was stored.



### The "System 2" Generation Flow
1.  **Trigger**: High logit entropy detected (the model realizes it is about to guess).
2.  **Search State**: The model pauses output and generates a **Latent Query Vector**.
3.  **Interrogation**: The query is compared against the MAC/KV memory in a loop.
4.  **Convergence**: The model iterates internally until the search result stabilizes.
5.  **Emission**: The retrieved vector is injected into the current layer to produce the "Needle."

---

## 3. Technical Implementation Strategy

### Updated MAC Layer (Pseudo-Code)
```python
class TitansSearchMAC(nn.Module):
    def forward(self, hidden_states):
        # 1. GENERATE SEARCH INTENT
        # Instead of f(h) ≈ h, we learn f(h) → Query_Vector
        latent_query = self.query_projector(hidden_states)

        # 2. SYSTEM 2 LOOP (Latent Reasoning)
        for _ in range(self.thinking_steps):
            # Differentiable search against internal weights
            search_result = self.memory_mlp.search(latent_query)
            # Refine the query based on the result's "sharpness" (Gradient Descent)
            latent_query = self.refine_query(latent_query, search_result)

        # 3. GATED INJECTION
        # Only inject if the search found something relevant
        gate = torch.sigmoid(self.relevance_head(search_result))
        return hidden_states + (search_result * gate)
```



## 4. Next Action Plan: "Forced Alignment"

### Experiment 7: Contrastive Memory Anchoring
* **Objective**: Link "Question Intent" to "High Surprise Updates."
* **Method**: During TTT (Test-Time Training), implement a **Contrastive Loss**.
* **Goal**: Minimize the mathematical distance between the Query(Question) and the Weights(Needle) specifically when the "Surprise" factor is high.

### Experiment 8: The Gradient Stop-Condition
* **Objective**: Establish dynamic "Thinking" time based on confidence.
* **Method**: Monitor the **MAC Gradient Magnitude** during the latent loop.
* **Logic**: Keep "Thinking" while the internal memory state is shifting. Only generate a token when the gradient flattens (convergence).

---

## 5. Technical Glossary
* **Differentiable Search**: Transitioning from a discrete lookup to an optimized latent query using internal gradients.
* **Wave Function Collapse**: The moment a high-entropy latent search commits to a single output token after reasoning.
* **Representation Drift**: The tendency for latent vectors to lose semantic meaning if not "pinned" to language during training.

---

**Status**: Active Research Phase 2  
**Current Task**: Implementation of Query Projector and Alignment Loss.
