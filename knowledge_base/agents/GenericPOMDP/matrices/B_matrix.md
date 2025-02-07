---
type: matrix_spec
id: B_matrix_001
matrix_type: transition
created: {{date}}
modified: {{date}}
tags: [matrix, transition, active-inference]
related_spaces: [s_space, pi_space]
---

# B-Matrix (Transition/Dynamics)

## Definition
The B-matrix defines state transition probabilities under different actions, representing P(s'|s,π) where s' is the next state, s is the current state, and π is the selected policy/action.

## Matrix Structure
```yaml
dimensions:
  rows: num_states        # From [[s_space]] (next state)
  cols: num_states       # From [[s_space]] (current state)
  depth: num_actions     # From [[pi_space]] (actions)
shape_constraints:
  - rows == cols  # Same state space for transitions
  - sum(rows) == 1.0  # Each column per action sums to 1
  - all_values >= 0  # Probability constraints
```

## Mathematical Form
```python
# Transition probability
P(s'|s,π) = B[s', s, π]

# Constraints
∀s,π: ∑_{s'} B[s',s,π] = 1  # Row-wise normalization per action
B[s',s,π] ≥ 0               # Non-negative probabilities
```

## Data Structure
```yaml
matrix_data:
  format: numpy.ndarray
  dtype: float32
  shape: [num_states, num_states, num_actions]
  initialization: identity_based  # Start with strong self-transitions
  storage: matrix_store/B_matrix.npy
```

## Visualization
```yaml
plot_type: multi_heatmap
colormap: "Blues"
title: "Transition Matrices (B) per Action"
xlabel: "Current State"
ylabel: "Next State"
slices: "Actions"
```

## Update Rules
- [[gradient_B]] - Learning transitions from experience
- [[structure_learning_B]] - Learning causal structure
- [[control_adaptation_B]] - Adapting to control dynamics

## Properties
### Markov Property
- Transitions depend only on current state
- Independent of history
- [[markov_chain]] properties

### Action Effects
- Different slices for each action
- May be sparse (impossible transitions)
- [[action_impact]] analysis

### Conservation
- Probability mass conservation
- [[transition_constraints]]
- [[physical_constraints]]

## Integration
- Used in [[state_prediction]]
- Affects [[policy_selection]]
- Links to [[action_model]]

## Examples
```python
# Example 2-state, 2-action B-matrix
B = np.array([
    # Action 1
    [[0.9, 0.1],  # Stay in state 1
     [0.2, 0.8]], # Stay in state 2
    # Action 2
    [[0.6, 0.4],  # Move to state 2
     [0.4, 0.6]]  # Move to state 1
])
```

## Learning Methods
- [[empirical_transitions]] - Learning from data
- [[causal_discovery]] - Structure learning
- [[inverse_dynamics]] - Learning from outcomes

## Computational Interface
```python
class TransitionMatrix:
    def __init__(self, num_states: int, num_actions: int):
        self.B = self._initialize_transitions(num_states, num_actions)
    
    def get_transition_probs(self, state: int, action: int) -> np.ndarray:
        """Get transition probabilities for state-action pair"""
        return self.B[:, state, action]
    
    def update_transitions(self, 
                         state: int, 
                         action: int, 
                         next_state: int,
                         learning_rate: float):
        """Update transition probabilities based on observation"""
        pass
```

## Related Components
- [[s_space]] - State space definition
- [[pi_space]] - Policy/action space
- [[dynamics_model]] - System dynamics
- [[control_theory]] - Control principles

## References
- [[markov_decision_process]]
- [[transition_learning]]
- [[control_theory]] 