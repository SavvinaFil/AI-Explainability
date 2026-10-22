# Tutorial: Explaining a Reinforcement-Learning grid agent

This guide shows how to explain a **reinforcement-learning control agent** for a
power grid with the toolbox. The running example is an agent whose policy network
maps a grid **observation** (state) to a set of **actions** — some continuous
(e.g. active/reactive power setpoints `P`, `Q` of static generators) and some
discrete (e.g. transformer **tap** positions).

The key idea: a policy network is just a **feedforward neural network with
multiple outputs**. The toolbox explains each output with SHAP — telling you
*which observation features drove each action*. Continuous setpoints and discrete
tap logits are treated identically; the only difference is how you read them.

> **What you need:** a trained policy network (or a function that calls it) and a
> table of **observations** (states) to explain. You do **not** need the recorded
> actions — SHAP queries the network itself. A few hundred representative states
> (from rollouts or operational logs) is plenty.

---

## 1. What gets explained

| Output kind | Example | What SHAP tells you |
| :--- | :--- | :--- |
| Continuous setpoint | `P_gen1`, `Q_gen1` | Which observations push the setpoint up or down. |
| Discrete action (logits) | `tap_0`, `tap_1`, `tap_2` | Which observations favour each tap position; `argmax` = the tap the agent picks. |

Each output becomes one entry in `result.shap_values` — a `(n_states, n_features)`
array. `output_labels` give them readable names.

---

## 2. Minimal example (PyTorch policy)

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ai_explainability as aie

# --- 1. A trained policy: observation(4) -> 5 action components -------------
# outputs: 0=P_gen1, 1=Q_gen1 (continuous); 2,3,4 = tap-position logits.
policy_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 5))
# (here untrained for illustration; use your trained weights)

OBS = ["line_load", "v_bus1", "v_bus2", "demand"]
ACTIONS = {"0": "P_gen1", "1": "Q_gen1", "2": "tap_0", "3": "tap_1", "4": "tap_2"}

# --- 2. A table of states to explain (from rollouts / logs) ----------------
states = pd.DataFrame(np.random.randn(200, 4), columns=OBS)

# --- 3. Explain every action component --------------------------------------
result = aie.explain(
    model=policy_net,
    data=states,
    analysis="tabular",
    model_type="rl_policy",          # alias of the feedforward explainer
    package="pytorch",
    explainer_type="deep",           # "kernel" (model-agnostic) / "deep" / "gradient"
    feature_names=OBS,
    output_labels=ACTIONS,
    background_size=100,             # reference states for the SHAP baseline
)

result.shap_values[0]   # SHAP for P_gen1: (200, 4)
result.shap_values[2]   # SHAP for tap position 0
result.to_dataframe()   # flat table across all actions
```

---

## 3. Explaining the *chosen* action

For the discrete tap head, the agent takes the `argmax` over the tap outputs.
`chosen_output` attributes, per state, the tap it actually selected:

```python
chosen_tap, chosen_shap = result.chosen_output(among=[2, 3, 4])

chosen_tap    # (200,)  -> output index of the selected tap for each state
chosen_shap   # (200,4) -> SHAP of the selected tap: "why did it pick this one here?"
```

`among=[2, 3, 4]` restricts the argmax to the discrete head, so continuous
setpoints (0, 1) are not mixed in. Continuous actions don't need this — the
output *is* the action.

---

## 3b. Fully discrete agents with several heads (P, Q, and tap all discrete)

The toolbox works for **any combination** of continuous and discrete outputs,
because it explains the network's output *columns* regardless of how they're
interpreted. The only rule is how an action maps to columns:

- a **continuous** action = **one** output column;
- a **discrete** action = a **group** of output columns (one logit per category);
  `argmax` over that group is the chosen category.

So if the agent chooses discrete P (say 3 levels), discrete Q (3 levels) and a
discrete tap (4 positions), the network outputs `3 + 3 + 4 = 10` logit columns.
Tell the toolbox which columns form each head and get the chosen category + its
attribution per head in one call:

```python
# column layout of the policy output, fixed by your wrapper:
#   0,1,2  -> P levels      3,4,5 -> Q levels      6,7,8,9 -> tap positions
ACTIONS = {
    "0": "P_low", "1": "P_mid", "2": "P_high",
    "3": "Q_low", "4": "Q_mid", "5": "Q_high",
    "6": "tap_0", "7": "tap_1", "8": "tap_2", "9": "tap_3",
}

result = aie.explain(
    model=policy_predict, data=states, analysis="tabular",
    model_type="rl_policy", package="callable", explainer_type="kernel",
    feature_names=OBS, output_labels=ACTIONS, background_size=100,
)

chosen = result.chosen_actions({
    "P":   [0, 1, 2],
    "Q":   [3, 4, 5],
    "tap": [6, 7, 8, 9],
})

chosen["P"]    # (idx_per_state, shap_per_state) — which P level, and why
chosen["Q"]    # same for Q
chosen["tap"]  # same for tap
```

`chosen_actions` just applies `chosen_output` to each head, so it scales to any
number of heads and any head sizes. A continuous head is simply a head with one
index (no argmax needed — the output is the action).

### Faster: explain only the action taken (`targets=`)

The run above attributes **every** output column and throws most of that work
away when you only care about the chosen logit per head. Pass the heads to
`explain()` instead, and only the selected output of each head is explained:

```python
heads = {"P": [0, 1, 2], "Q": [3, 4, 5], "tap": [6, 7, 8, 9]}

result = aie.explain(
    model=policy_predict, data=states, analysis="tabular",
    model_type="rl_policy", package="callable", explainer_type="kernel",
    feature_names=OBS, background_size=100,
    targets=heads,              # or targets="argmax" for a single head
)

result.shap_values["P"]           # (n_states, n_features) — SHAP of the chosen P
result.chosen_index["P"]          # which P output was chosen, per state
chosen = result.chosen_actions(heads)   # same call as before, no recomputation
```

The argmax of each head is computed once on your states, and SHAP then runs on
a wrapped model that returns only the chosen logits — 3 outputs instead of 10
here (490 → 35 for a 35-head policy). The choice is fixed per state, so the
SHAP values are identical to the full run followed by `chosen_actions`.

For a cheaper overview, sum heads into families:

```python
result = aie.explain(..., targets=heads,
                     targets_reduce="sum", target_families={"PQ": ["P", "Q"]})
sorted(result.shap_values)   # ['PQ', 'tap'] — heads not in a family stay separate
```

`targets=` works with the `kernel`, `gradient` and `deep` backends for
feedforward / policy models, and cannot be combined with `target_index`.

---

## 4. If your policy isn't a plain `nn.Module`

RL frameworks (Maze, Stable-Baselines3, RLlib, custom) often wrap the network so
calling it returns a tuple/dict of heads, a distribution object, or needs a
`.predict()` call. Wrap it in a small function that maps a **batch of
observations** to a **2D array of the action-components you want explained**, and
pass that function as `model` with `package="callable"`:

```python
def policy_predict(obs_batch):
    obs = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32)
    with torch.no_grad():
        cont, tap_logits = my_agent.policy(obs)     # however your agent exposes heads
    # Concatenate the heads into one (n, n_actions) array, in a fixed column order.
    return torch.cat([cont, tap_logits], dim=1).numpy()

result = aie.explain(
    model=policy_predict,            # plain callable -> framework-agnostic
    data=states,
    analysis="tabular",
    model_type="rl_policy",
    package="callable",
    explainer_type="kernel",         # callable models support KernelSHAP
    feature_names=OBS,
    output_labels=ACTIONS,
    background_size=100,
)
```

The wrapper is the only integration code that needs to be written. It also fixes
the **column order**, which is what `target_index` and `output_labels` refer to.

- For a **continuous policy**, return the action **mean** (what the agent
  would deploy).
- For **discrete heads**, return the **logits** (pre-softmax); `argmax` matches the
  action taken, and `chosen_output` then attributes it.

With a plain callable only **KernelSHAP** is available (it treats the policy as a
black box). To use `deep`/`gradient`, pass the underlying `nn.Module`/Keras model
instead.

---

## 5. Choosing the SHAP backend

| `explainer_type` | Needs | Good for |
| :--- | :--- | :--- |
| `kernel` | any callable + a background set | Framework-agnostic; always works; slower (cost ∝ `kernel_nsamples`). |
| `deep` | a torch/keras module | Fast, exact-ish for deep nets; support varies by version. |
| `gradient` | a torch/keras module | Fast gradient-based attributions. |

Start with `kernel` to get results, then switch to `deep`/`gradient` for speed if
you can pass the raw network.

---

## 6. Saving a report

By default an in-memory run writes nothing. For an RL policy with many action
outputs, the **kernel-free HTML report** is recommended — it renders directly
with matplotlib (no Jupyter kernel, so it can't hit the kernel/zmq crashes seen
on some Windows setups) and gives one section per action:

```python
result.save_report_html(
    "output/grid_agent_report.html",
    output_labels=labels,   # names each action logit, e.g. "P_gen1=L0"
    max_outputs=None,       # cap sections if the policy has many logits
)
```

See [`output/report_rl_agent_example.html`](../../../output/report_rl_agent_example.html)
for a sample of this report on the all-discrete structure.

The Excel audit (one sheet per action) and the Jupyter notebook are also
available:

```python
result = aie.explain(..., save_excel=True, generate_notebook=True, output_dir="output/")
# or, after the fact:
result.save_excel("grid_agent_audit.xlsx")
result.save_notebook("grid_agent_report.ipynb")   # ready-to-run; execute_notebook=True to pre-render
```
