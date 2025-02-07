---
type: mathematical_concept
id: policy_selection_001
created: 2024-02-05
modified: 2024-02-05
tags: [mathematics, active-inference, policy-selection, decision-making]
aliases: [action-selection, policy-inference]
---

# Policy Selection

## Mathematical Definition

Policy selection in Active Inference is based on the softmax of negative expected free energy:

$P(\pi) = \sigma(-\gamma G(\pi))$

where:
- $G(\pi)$ is the [[expected_free_energy]] for policy $\pi$
- $\gamma$ is the precision parameter (inverse temperature)
- $\sigma$ is the softmax function

## Components

### Expected Free Energy
- Future-oriented evaluation
- [[epistemic_value]]
- [[pragmatic_value]]

### Policy Space
- Available action sequences
- [[E_matrix]] definition
- [[action_constraints]]

### Selection Mechanism
- Softmax transformation
- [[precision_parameter]]
- [[exploration_exploitation]]

## Implementation

```python
def select_policy(
    A: np.ndarray,           # Observation model from [[A_matrix]]
    B: np.ndarray,           # Transition model from [[B_matrix]]
    C: np.ndarray,           # Preferences from [[C_matrix]]
    E: np.ndarray,           # Policies from [[E_matrix]]
    beliefs: np.ndarray,     # Current beliefs Q(s)
    temperature: float = 1.0  # Softmax temperature
) -> Tuple[int, np.ndarray]:
    """
    Select action using Active Inference.
    
    Args:
        A: Observation likelihood matrix P(o|s)
        B: State transition matrix P(s'|s,a)
        C: Preference matrix over observations
        E: Policy matrix defining action sequences
        beliefs: Current belief distribution Q(s)
        temperature: Softmax temperature parameter
        
    Returns:
        Tuple of (selected action index, policy probabilities)
    """
    # Compute expected free energy for each policy
    expected_free_energies = np.zeros(len(E))
    
    for i, policy in enumerate(E):
        expected_free_energies[i] = compute_expected_free_energy(
            A=A,
            B=B,
            C=C,
            beliefs=beliefs,
            action=policy[0]  # Consider first action of policy
        )
    
    # Convert to policy probabilities using softmax
    policy_probs = softmax(-expected_free_energies / temperature)
    
    # Sample action from policy distribution
    selected_policy = np.random.choice(len(policy_probs), p=policy_probs)
    selected_action = E[selected_policy][0]  # First action of selected policy
    
    return selected_action, policy_probs
```

## Usage

Policy selection is used in:
- [[action_selection]] - Choosing actions
- [[planning]] - Multi-step planning
- [[active_inference_loop]] - Core decision step

## Properties

### Mathematical Properties
- [[optimality_guarantees]]
- [[exploration_control]]
- [[policy_convergence]]

### Computational Properties
- [[sampling_efficiency]]
- [[parallelization]]
- [[scalability]]

## Variants

### Single-Step
- [[greedy_selection]]
- [[epsilon_greedy]]
- [[thompson_sampling]]

### Multi-Step
- [[tree_search]]
- [[monte_carlo]]
- [[trajectory_optimization]]

### Hierarchical
- [[option_framework]]
- [[hierarchical_policies]]
- [[abstraction_levels]]

## Related Concepts
- [[decision_theory]]
- [[reinforcement_learning]]
- [[optimal_control]]

## Implementation Details

### Numerical Considerations
- [[temperature_annealing]]
- [[precision_adaptation]]
- [[numerical_stability]]

### Optimization
- [[policy_caching]]
- [[batch_processing]]
- [[pruning_strategies]]

## References
- [[friston_policies]] - Policy theory
- [[active_inference_control]] - Control applications
- [[implementation_examples]] - Code examples 