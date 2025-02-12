---
title: Information Theory
type: knowledge_base
status: stable
created: 2024-03-20
tags:
  - mathematics
  - information
  - computation
  - probability
semantic_relations:
  - type: implements
    links: [[mathematical_foundations]]
  - type: extends
    links: [[probability_theory]]
  - type: related
    links: 
      - [[information_gain]]
      - [[information_geometry]]
      - [[active_inference]]
      - [[free_energy_principle]]
---

# Information Theory

Information theory provides the mathematical framework for quantifying, storing, and communicating information. Within the active inference framework, it formalizes the principles of uncertainty, surprise, and information gain that drive cognitive processes through prediction error minimization.

## Core Principles

### Entropy
1. **Shannon Entropy**
   ```math
   H(X) = -∑ P(x)log₂P(x)
   ```
   where:
   - H(X) is entropy
   - P(x) is probability distribution
   - log₂ gives bits of information

2. **Differential Entropy**
   ```math
   h(X) = -∫ p(x)log p(x)dx
   ```
   where:
   - h(X) is differential entropy
   - p(x) is probability density

### Mutual Information
1. **Discrete Form**
   ```math
   I(X;Y) = ∑∑ P(x,y)log₂(P(x,y)/P(x)P(y))
   ```
   where:
   - I(X;Y) is mutual information
   - P(x,y) is joint distribution
   - P(x), P(y) are marginals

2. **Continuous Form**
   ```math
   I(X;Y) = ∫∫ p(x,y)log(p(x,y)/p(x)p(y))dxdy
   ```
   where:
   - p(x,y) is joint density
   - p(x), p(y) are marginal densities

## Information Measures

### Divergence Metrics
1. **Kullback-Leibler Divergence**
   ```math
   D_KL(P||Q) = ∑ P(x)log(P(x)/Q(x))
   ```
   where:
   - D_KL is KL divergence
   - P, Q are distributions

2. **Jensen-Shannon Divergence**
   ```math
   JSD(P||Q) = ½D_KL(P||M) + ½D_KL(Q||M)
   ```
   where:
   - M = ½(P + Q)
   - JSD is symmetric measure

### Information Flow
1. **Transfer Entropy**
   ```math
   T_Y→X = ∑ p(x_{t+1},x_t,y_t)log(p(x_{t+1}|x_t,y_t)/p(x_{t+1}|x_t))
   ```
   where:
   - T_Y→X is transfer entropy
   - p(x_{t+1}|x_t,y_t) is transition probability

2. **Directed Information**
   ```math
   I(X→Y) = ∑ᵢ I(X^i;Yᵢ|Y^{i-1})
   ```
   where:
   - I(X→Y) is directed information
   - X^i, Y^{i-1} are past sequences

## Active Inference Connection

### Free Energy Decomposition
1. **Variational Free Energy**
   ```math
   F = D_KL[q(s)||p(s|o)] - log p(o)
   ```
   where:
   - F is free energy
   - q(s) is approximate posterior
   - p(s|o) is true posterior

2. **Expected Free Energy**
   ```math
   G = D_KL[q(o|π)||p(o)] + E_q[H(o|s,π)]
   ```
   where:
   - G is expected free energy
   - π is policy
   - H is conditional entropy

### Information Gain
1. **Epistemic Value**
   ```math
   EV = E_q[D_KL[q(s|o,π)||q(s|π)]]
   ```
   where:
   - EV is epistemic value
   - q(s|o,π) is posterior
   - q(s|π) is prior

2. **Salience**
   ```math
   W = -∇_oF = ∇_o log p(o|s)
   ```
   where:
   - W is precision-weighted prediction error
   - p(o|s) is likelihood

## Applications

### Neural Coding
1. **Efficient Coding**
   - Rate coding
   - Population coding
   - Sparse coding
   - Predictive coding

2. **Neural Information**
   - Spike train entropy
   - Neural efficiency
   - Code redundancy
   - Error correction

### Cognitive Processes
1. **Perception**
   - Feature extraction
   - Pattern recognition
   - Information integration
   - Uncertainty reduction

2. **Learning**
   - Information acquisition
   - Model selection
   - Belief updating
   - Knowledge compression

## Future Directions

1. **Theoretical Extensions**
   - Quantum information
   - Non-equilibrium processes
   - Complex networks
   - Causal inference

2. **Applications**
   - Neural architectures
   - Cognitive models
   - Learning systems
   - Clinical interventions

## Related Concepts
- [[information_gain]]
- [[information_geometry]]
- [[active_inference]]
- [[free_energy_principle]]
- [[bayesian_inference]]

## References
- [[shannon_1948]] - "A Mathematical Theory of Communication"
- [[mackay_2003]] - "Information Theory, Inference, and Learning Algorithms"
- [[cover_thomas_2006]] - "Elements of Information Theory"
- [[friston_2010]] - "The Free-Energy Principle"