# Titans MAC (Memory as Context) - Investigation & Implementation

This project implements a **Memory as Context (MAC)** layer, inspired by the Google DeepMind "Titans" architecture, integrated into various LLMs (Gemma, Qwen) to enable "infinite context" through neural memory.

## 🧠 The Core Concept: Infinite Context via Neural Memory

### The Problem: Attention Bottleneck
Standard Transformers suffer from fundamental limitations:
1. **Quadratic Cost**: Attention grows with $O(N^2)$ where N = context length
2. **Hard Limit**: Once tokens slide out of the context window (e.g., 8k), they are lost forever
3. **Memory Constraint**: Longer contexts require massive GPU memory

### The Titans Solution
Instead of storing *raw tokens* of the past, we store a **compressed neural representation** in a separate MLP module:

| Approach | Storage | Mechanism | Location |
|----------|---------|-----------|----------|
| **Traditional RAG** | Text chunks | Vector database (external) | Outside model |
| **Titans MAC** | Neural weights | MLP parameters (internal) | Inside model |

### Test-Time Training (TTT)
The core innovation: **The model learns while generating**. Unlike frozen LLMs, our model continuously updates its memory based on "surprise" - high reconstruction error signals novel information worth remembering.

## 🏗️ Architecture Overview

### The MAC Layer (`TitansMACLayer`)
```python
class TitansMACLayer(nn.Module):
    - MemoryMLP: 2-layer MLP (hidden_size → 4*hidden_size → hidden_size)
    - LayerNorm: Stabilizes the memory output
    - Surprise Threshold: Gates memory updates
```

**Integration Point**: Injected at layer `N // 2` (middle of transformer) via forward pre-hooks

**Algorithm**:
1. Forward pass through transformer layer
2. MAC attempts to reconstruct hidden states
3. Compute reconstruction loss (MSE)
4. If `loss > threshold`: Update memory weights via SGD
5. Inject memory output residually into the stream

### Gradient Management
The trickiest part: Running TTT during inference.

**Problem**: `model.generate()` runs in `torch.no_grad()` mode
**Solution**: Wrap MAC forward in `torch.enable_grad()` and manually manage gradient graphs

```python
with torch.enable_grad():
    inp = current_hidden_states.detach().requires_grad_(True)
    memory_output = self.memory_mlp(inp)
    loss = F.mse_loss(memory_output, inp)
    if loss > threshold:
        self.memory_mlp.update_memory(inp, loss)  # SGD update
```

## 🔬 Experimental Investigation

### Models Tested

| Model | Size | Status | Notes |
|-------|------|--------|-------|
| Qwen3-4B-Thinking-2507-FP8 | 4B | ❌ Failed | FP8 + ROCm incompatibility, hung during generation |
| Qwen3-4B-Thinking-2507 | 4B | ❌ Failed | Thinking model + long output caused GPU crash (loss → inf) |
| **Qwen3-4B-Instruct-2507** | 4B | ✅ Success | Stable, all tests passed |
| Qwen2.5-0.5B | 0.5B | ✅ Success | Used for initial validation |

### Critical Issues & Solutions

#### Issue 1: FP8 Models Load on CPU
**Problem**: `Qwen3-4B-Thinking-FP8` loaded on CPU despite `.to(device)` call

**Error**:
```
You have loaded an FP8 model on CPU and have a CUDA device available
```

**Root Cause**: FP8 models require `device_map="cuda"` in `from_pretrained()`, not `.to(device)` afterwards

**Fix**:
```python
# OLD (broken):
model = AutoModelForCausalLM.from_pretrained(name).to(device)

# NEW (works):
model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda")
```

#### Issue 2: Dtype Mismatch
**Problem**: RuntimeError during forward pass
```
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

**Root Cause**: Model is BFloat16 but MAC layer was default Float32

**Fix**:
```python
model_dtype = next(self.model.parameters()).dtype
self.mac_layer = TitansMACLayer(hidden_size).to(device=device, dtype=model_dtype)
```

#### Issue 3: Model Hanging / GPU Crash
**Problem**: Thinking model caused GPU hardware exception during generation

**Root Cause**: Learning rate too high (0.01) → gradient explosion → loss → inf → GPU crash

**Fix**: Reduced learning rate from 0.01 back to 1e-4

### Parameter Tuning Journey

#### Initial Parameters (from reference implementation)
```python
learning_rate: 1e-4
injection_scale: 0.01  # Very conservative
surprise_threshold: 50.0
```

#### Tuning Iterations

| Iteration | LR | Injection | Threshold | Result |
|-----------|----|----------|----------|--------|
| **Start** | 1e-4 | 0.01 | 50.0 | Too weak, no effect |
| **Aggressive** | 0.01 | 1.0 | 50.0 | GPU crash (loss → inf) |
| **Balanced** | 1e-4 | 0.1 | 50.0 | No updates in Phase 2 |
| **Final** | 1e-4 | 0.1 | 25.0 | ✅ Updates in learning phase |

**Final Configuration**:
```python
learning_rate: 1e-4          # Stable, no explosion
injection_scale: 0.1         # Noticeable but not disruptive
surprise_threshold: 25.0     # Captures novel content in 1024-token chunks
memory_reset: True           # Reset before learning phase
```

## 📊 Testing Methodology

### The Needle-in-a-Haystack Test

**Goal**: Test if MAC can retrieve a fact (the "needle") from a long document (the "haystack")

**Need**: "The giant's favorite color is MAGENTA."
**Query**: "What is the giant's favorite color?"

### Test Phases

#### Phase 0: Control Test (Sanity Check)
**Purpose**: Verify model can retrieve from short context

**Method**:
```python
prompt = "The giant's favorite color is MAGENTA. What is the giant's favorite color?"
response = model.generate(prompt)
```

**Result**: ✅ PASS - Model answers "MAGENTA"

#### Phase 1: Construct Haystack
**Purpose**: Create a long document with the needle embedded

**Method**: Use Greek mythology text (public domain) with needle inserted in the middle
```python
context = greek_mythology_text  # ~1500 tokens
needle = "The giant's favorite color is MAGENTA."
haystack = context_part1 + needle + context_part2  # ~7870 tokens
```

#### Phase 2: Feed Haystack (Learning Phase)
**Purpose**: Feed the long context to trigger MAC learning

**Method** (WRONG - doesn't work):
```python
# Feed in chunks
for chunk in chunks:
    model(chunk)  # Process but don't store

# Later query
response = model.generate(query)  # Separate forward pass - no context!
```

**Issue**: This tests if the model can "remember" from a previous forward pass, which it can't without MAC. But it's also wrong for baseline testing.

#### Phase 3: MAC Retrieval Test
**Purpose**: Test if MAC learned the needle

**Current Method**:
1. Reset MAC memory to fresh state
2. Feed haystack in chunks (triggers MAC updates on "surprising" content)
3. Query with just the question text
4. Check if response contains "MAGENTA"

**Result**: ❌ FAIL - Even with MAC updates, retrieval doesn't work

#### Phase 4: Baseline Test (Corrected)
**Purpose**: Verify model CAN retrieve when context is IN the prompt

**Method** (CORRECT):
```python
prompt_with_context = haystack + "\n\n" + query
response = model.generate(prompt_with_context)  # Context included!
```

**Result**: ✅ PASS - Model successfully retrieves "MAGENTA" from 7870 tokens

## 🎯 Key Findings

### 1. Baseline Capability is Excellent
**Discovery**: Qwen3-4B-Instruct can retrieve from ~8500 tokens when context is in the prompt

| Context Length | Result |
|----------------|--------|
| 860 tokens | ✅ PASS |
| 1719 tokens | ✅ PASS |
| 4296 tokens | ✅ PASS |
| 6873 tokens | ✅ PASS |
| **8591 tokens** | ✅ PASS |

**Implication**: The model doesn't need MAC for < 8k contexts. MAC would only help beyond ~32k tokens.

### 2. MAC Learning Behavior

**Observation**: MAC learns VERY quickly - often in a single update

```
Phase 0 (fresh memory):
  Token 1: Loss 684.00 → SURPRISE! → Memory updated
  Token 2: Loss 0.66 → No surprise (already learned!)
  Token 3: Loss 1.00 → No surprise
```

**Issue**: The MLP (2 layers, 5120 hidden units) overfits almost instantly due to:
- Large network capacity
- Simple identity reconstruction task
- High learning rate

### 3. Surprise Threshold Sensitivity

**Tested Values**:
- `500.0`: No updates at all (too high)
- `100.0`: No updates in Phase 2 (too high)
- `50.0`: No updates in Phase 2 (too high)
- **`25.0`**: ✅ 5 updates during Phase 2 (works!)

**Why so low?** The MSE loss is averaged over:
- 1024 tokens per chunk ×
- 2560 dimensions = 2.6 million values

Even with fresh random weights, the average loss is only 20-50. The needle's contribution is diluted.

### 4. Memory Updates Don't Enable Retrieval

**The Problem**: Even with 5 surprise updates during learning, Phase 3 retrieval fails

**Root Cause Analysis**:

1. **Reconstruction ≠ Association**:
   - MAC learns to reconstruct hidden states
   - It learns the AVERAGE of all seen states
   - "MAGENTA" is < 0.1% of training data

2. **Query ≠ Answer States**:
   - Query: "What is the giant's favorite color?"
   - Answer: "The giant's favorite color is MAGENTA."
   - These produce very different hidden states!

3. **Injection Doesn't Help**:
   - MAC injects learned patterns into query states
   - But query states don't match answer states
   - So injection doesn't trigger retrieval

### 5. Fundamental Architecture Limitation

**Conclusion**: The reconstruction-based MAC approach is fundamentally unsuited for **retrieval tasks**.

**Why?**
- Reconstruction learning = "Learn to output what you saw"
- Retrieval requires = "Learn to associate question with answer"
- These are different objectives!

**Where MAC WOULD Work**:
- **Continuation tasks**: "Here's a document... continue it in the same style"
- **Pattern completion**: "Complete this code/math problem"
- **Style transfer**: Maintain voice/tone across long documents

**Where MAC DOESN'T Work**:
- **Fact retrieval**: "What was the specific fact I told you 10k tokens ago?"
- **Key-value storage**: Storing discrete information for later lookup

## 🛠️ Implementation Details

### File Structure
```
miras-test/
├── gemma/
│   ├── gemma_mac.py           # Gemma-specific implementation
│   ├── interact_mac.py        # Interactive CLI
│   └── needle_test.py         # Testing script
├── qwen3_0.6B/               # Qwen 0.5B (working)
│   ├── qwen_mac.py
│   └── needle_test.py
├── qwen3_4b/                 # Qwen 4B (main focus)
│   ├── qwen_mac.py            # Core implementation
│   ├── needle_test.py         # Needle-in-haystack test
│   ├── find_baseline_natural.py    # Baseline capability test
│   ├── find_baseline_correct.py    # Fixed baseline test
│   └── test_simple.py        # Simple generation test
└── README.md                 # This file
```

### Key Implementation Insights

#### 1. Forward Hook Registration
```python
def _register_hooks(self):
    def pre_hook(module, input):
        hidden_states = input[0]
        mem_out = self.mac_layer(hidden_states)

        self.total_tokens_processed += hidden_states.shape[1]

        if self.total_tokens_processed > self.start_memory_injection_threshold:
            # Inject memory into stream
            combined = hidden_states + 0.1 * mem_out
            return (combined, *input[1:])
        else:
            # Before threshold: Only learn, don't inject
            return input

    layer = self.layers_list[self.insertion_layer_idx]
    layer.register_forward_pre_hook(pre_hook)
```

#### 2. Memory Reset Functionality
```python
def reset_memory(self):
    """Reset memory weights to initial random state"""
    for layer in self.layers:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
```

**Critical for testing**: Allows fresh learning between test phases

#### 3. Proper Baseline Testing
```python
# WRONG (what we initially did):
model.feed_context(context)
response = model.generate(query)  # Context is forgotten!

# CORRECT (what we fixed it to):
prompt_with_context = context + "\n\n" + query
response = model.generate(prompt_with_context)  # Context included!
```

## 📈 Performance Summary

### Test Results (Final Configuration)

```
--- Phase 0: Control Test ---
✅ PASSED - Model retrieves "MAGENTA" from short context

--- Phase 2: Feeding Haystack ---
📊 MAC activated, 5 surprise updates:
  - Chunk 2: Loss 30.62 → Update
  - Chunk 3: Loss 26.88 → Update
  - Chunk 4: Loss 26.25 → Update
  - Chunk 5: Loss 34.25 → Update
  - Chunk 6: Loss 48.75 → Update

--- Phase 3: MAC Retrieval ---
❌ FAILED - Response: "There isn't a universally known giant's favorite color..."

--- Phase 4: Baseline (Context in Prompt) ---
✅ PASSED - Model retrieves "MAGENTA" from 7870-token context
```

### What We Proved
1. ✅ MAC can learn from streaming data (surprise mechanism works)
2. ✅ Parameters can be tuned for stable operation
3. ✅ Baseline models have excellent long-context capability (~8k tokens)
4. ❌ Reconstruction-based MAC doesn't enable fact retrieval

### What We Didn't Prove
- Whether MAC would work for > 32k token contexts (beyond model native capability)
- Whether different architectures (key-value memory) would work better
- Whether MAC helps with continuation/style tasks vs. retrieval tasks

## 🔮 Future Directions

### Alternative Architectures to Try

1. **Key-Value Memory**:
   ```python
   class KeyValueMemory:
       def store(self, key_hidden_states, value_hidden_states):
           self.keys[key] = value

       def retrieve(self, query_hidden_state):
           similarity = cosine_sim(query, self.keys)
           return self.values[argmax(similarity)]
   ```

2. **Explicit Text Storage**:
   ```python
   class TextMemory:
       def store(self, text, hidden_state):
           self.memory_vectors.append(hidden_state)
           self.memory_text.append(text)

       def retrieve(self, query):
           best_match = argmax(cosine_sim(query, self.memory_vectors))
           return self.memory_text[best_match]
   ```

3. **Hybrid Approach**:
   - Use reconstruction for pattern learning (continuation tasks)
   - Use key-value for fact retrieval (QA tasks)

### Testing Improvements

1. **Longer Contexts**: Test at 50k, 100k, 200k tokens to find actual limit
2. **Different Tasks**:
   - Code continuation (MAC should help)
   - Document summarization (MAC should help)
   - Multi-turn conversation (MAC should help)
3. **Needle Variations**:
   - Multiple needles
   - Needles at different positions
   - Needles with different "surprise" levels

### Parameter Optimization

Current settings are hand-tuned. Could use:
- Hyperparameter search (Optuna, Ray Tune)
- Automated threshold adaptation
- Dynamic learning rate scheduling

## 💡 Lessons Learned

### For LLM Architecture Research
1. **Test methodology matters**: Wrong baseline tests led to confusion
2. **Task-appropriate architecture**: Reconstruction ≠ Retrieval
3. **Overfitting is quick**: Large MLPs learn identity mapping instantly
4. **Threshold sensitivity**: Small changes (50→25) dramatically affect behavior

### For Debugging ML Systems
1. **Add extensive logging**: Debug messages revealed the learning pattern
2. **Test incrementally**: Each parameter in isolation
3. **Verify assumptions**: The MAC WAS running, just not learning what we expected
4. **Monitor GPU health**: ROCm exceptions caught early prevented damage

### For Reproducibility
1. **Document everything**: Every parameter, every change
2. **Version control**: Git history shows evolution
3. **Environment matters**: ROCm vs CUDA, FP8 vs BF16
4. **Random seeds**: Not set in this investigation (should be)

## 📚 References

- **Titans Paper**: "Titans: Learning to Memorize at Test Time" (Google DeepMind)
- **TTT Concept**: "Test-Time Training" (Grave et al., 2020)
- **Qwen Models**: https://huggingface.co/Qwen
- **ROCm**: AMD's CUDA alternative for GPU acceleration

## 🤝 Contributing

This is a research investigation. Areas for contribution:
- Alternative MAC architectures
- Better baseline tests
- Longer context evaluation
- Different model backends (Llama, Mistral, etc.)

## 📄 License

This code is research/educational. Models are subject to their respective licenses (Qwen: Apache 2.0).

## 🙏 Acknowledgments

- Google DeepMind for the Titans architecture
- Qwen team for the excellent models
- AMD for ROCm support (when it works!)

---

**Last Updated**: 2025-12-27
**Status**: Investigation Complete - MAC not suitable for retrieval tasks as implemented
**Next Steps**: Try key-value memory architecture or accept 8k token limit for this model
