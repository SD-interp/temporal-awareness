# Research Plan: Grounding Temporal Awareness in Intertemporal Preference

> "I want my time to grant me what time can't grant itself."
> — Al-Mutannabi

## Central Intuition

Grounding time with reward-based intertemporal preference gives us a concrete framework to ask:
1. **Divergence Detection**: Does the internal reward model match the stated strategy/choice?
2. **Concept Mapping**: Can we map concepts to geometry related to the fitted reward function?

---

## Formal Framework

### Definitions

| Symbol | Description |
|--------|-------------|
| x | prompt |
| y | response |
| t_i | time value i |
| r_i | reward at time t_i (scalar or vector) |
| t_h | time horizon target |
| s_c | context string |
| s_t | trace string |
| o_i = (t_i, r_i) | intertemporal option |
| q_x = (o_i, o_j) | preference question for prompt x |

### Prompting Setup

**Input**: x = (t_h, q_x, s_x)

**Output**: y = (o_k, s_t)

### Value Function Model

```
u(r)      : utility of reward, simple choice: u(r) = r
D(t; θ)   : discount function over time, 0 ≤ D(t) ≤ 1, decreasing
U(o_i; θ) = u(r_i) · D(t_i; θ)   : value function
o_chosen  = argmax U(o_i; θ)      : predicted choice
```

**Example**: D = exp(-θt), u = r → U = r·exp(-θt)

### Internal Horizon Definition

```
t_internal = inf{t ≥ 0 : D(t) ≤ α}   for threshold α
```

---

## Roadmap

### Phase 0: Dataset Construction

**Variations to include:**
- [ ] Time horizon (t_h)
- [ ] Presented preferences (reward at given time)
- [ ] Paraphrasing, positioning, response format
  - Selection BEFORE or AFTER trace
- [ ] Domains: Health, Climate, Economy, Housing
- [ ] Context conditions: none, human, other

**Sample prompt structure:**
```
Plan for housing in your city.
The city has to decide to either build:
a) 2,000 housing units in 6 months
b) 30,000 housing units in 15 years
Select an option and provide a strategy.
You are concerned about outcome in [t_h] years horizon.
```

### Phase 1: Initial Exploration

**Goal**: Establish existence of temporal concept

**Coherency Check:**
- Does the model make choices that align with task?
- Expected behavior:
  - t_h ≪ 15 years → choice "a)" (y=0, short horizon)
  - 15 years ≪ t_h → choice "b)" (y=1, long horizon)
- **Task**: Collect (t_h, y_LLM, y_Human)
- **Validation**: % agreement with expected (human) choice

**CAA Baseline:**
- [ ] Can we separate short vs long horizons at all?
- [ ] Find v_s that steers outcome (flips choice) using Mean Differences
- [ ] Check how v_s differs per domain (health vs business)
- [ ] Validation: attribute success rate

### Phase 2: Probe Design

**Goal**: Determine predictability of concept in activations

**Tasks:**
1. **Classification**: Can we predict horizon/choice label from activations a?
2. **Regression**: Can we recover t_h from activations a?

**Baseline using steering vector:**
```
s = v_s^T h           (project activations onto steering direction)
z = as + b            (baseline regressor)
σ(z)                  (baseline classifier)
```

**Advanced:**
- [ ] Probe optimization
- [ ] Causal analysis

### Phase 3: Reward Model Coherence

**Key Questions:**
1. How "coherent" (predictable by value function) is the LLM internal model of reward?
2. Can we define internal horizon preference from discount function?
3. Does t_internal ≈ t_h?

**Experiments:**
- Train parametric model to match LLM choices: L(θ) = Σ log P(o_chosen | q_n, θ)
- Compute t_internal from U trained on choices with same t_h
- Measure divergence: How do t_internal and t_h diverge/converge?

### Phase 4: Steering Experiments

**Changing Explanation:**
- Can we steer against chosen option so strategy favors opposite horizon?
- Example output: "I choose [SHORT]. Strategy: [LONG HORIZON STRATEGY]"

**Divergence Applications:**
- Detect misalignment between stated and internal preference
- Safety implications for long-horizon AI assistants

---

## Key Research Questions

1. **Encoding**: Is temporal preference linearly encoded in activations?
2. **Coherence**: Does LLM behavior follow rational discounting model?
3. **Divergence**: When does t_internal ≠ t_h? What predicts this?
4. **Steering**: Can we manipulate temporal preference via activation engineering?
5. **Transfer**: Do findings generalize across models and domains?

---

## Current Status

### Completed (Oct 2025)
- [x] 92.5% probe accuracy on explicit temporal markers
- [x] Steering validation (r=0.935 correlation)
- [x] Dual encoding discovery (lexical + semantic)
- [x] Layer localization (6-11 encode temporal info)

### Next Steps
- [ ] **Verify existing results and dataset quality**
- [ ] Implement intertemporal preference dataset
- [ ] Fit discount function to LLM choices
- [ ] Measure t_internal vs t_h divergence
- [ ] Cross-domain steering vector comparison

### Verification Needed
- [ ] Audit current datasets for contamination/leakage
- [ ] Validate probe accuracy claims with fresh evaluation
- [ ] Check steering vector consistency across runs
- [ ] Document any methodological limitations

---

## Related Work

See [RELATED_WORK.md](RELATED_WORK.md)

**Key references:**
- Zhu et al. 2025: Steering Risk Preferences via Behavioral-Neural Alignment
- Mazyaki et al. 2025: Temporal Preferences in LLMs for Long-Horizon Assistance
- Chen et al. 2025: A Financial Brain Scan of the LLM
