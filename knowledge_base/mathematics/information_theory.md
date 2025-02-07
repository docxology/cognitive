# Information Theory in Cognitive Modeling

---
type: mathematical_concept
id: information_theory_001
created: 2024-02-05
modified: 2024-02-06
tags: [mathematics, information-theory, entropy, inference]
aliases: [information-theory, entropy-theory]
semantic_relations:
  - type: implements
    links: 
      - [[../../docs/research/research_documentation_index|Research Documentation]]
      - [[free_energy_principle]]
  - type: uses
    links:
      - [[variational_methods]]
      - [[statistical_foundations]]
  - type: documented_by
    links:
      - [[../../docs/guides/implementation_guides_index|Implementation Guides]]
      - [[../../docs/api/api_documentation_index|API Documentation]]
---

## Overview

Information theory provides the mathematical foundation for quantifying uncertainty, information gain, and surprise in cognitive modeling. This document outlines key information-theoretic concepts and their applications in active inference and predictive processing.

## Core Concepts

### Entropy Measures
```python
class EntropyMeasures:
    """
    Implementation of various entropy measures.
    
    Theory:
        - [[shannon_entropy]]
        - [[differential_entropy]]
        - [[renyi_entropy]]
    """
    @staticmethod
    def shannon_entropy(p: np.ndarray, base: float = np.e) -> float:
        """Compute Shannon entropy."""
        # Handle numerical stability
        p = np.clip(p, 1e-10, 1.0)
        return -np.sum(p * np.log(p) / np.log(base))
    
    @staticmethod
    def differential_entropy(p: Distribution) -> float:
        """Compute differential entropy for continuous distributions."""
        return -p.expect(lambda x: np.log(p.pdf(x)))
    
    @staticmethod
    def renyi_entropy(p: np.ndarray, alpha: float = 2.0) -> float:
        """Compute Rényi entropy of order alpha."""
        return np.log(np.sum(p**alpha)) / (1 - alpha)
```

### Divergence Measures
```python
class DivergenceMeasures:
    """
    Implementation of information divergences.
    
    Theory:
        - [[kullback_leibler]]
        - [[jensen_shannon]]
        - [[f_divergences]]
    """
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence."""
        # Numerical stability
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return np.sum(p * (np.log(p) - np.log(q)))
    
    @staticmethod
    def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence."""
        m = 0.5 * (p + q)
        return 0.5 * (DivergenceMeasures.kl_divergence(p, m) + 
                     DivergenceMeasures.kl_divergence(q, m))
```

### Mutual Information
```python
class MutualInformation:
    """
    Mutual information computation.
    
    Theory:
        - [[mutual_information]]
        - [[conditional_mutual_information]]
        - [[information_bottleneck]]
    """
    @staticmethod
    def compute_mi(joint_dist: np.ndarray) -> float:
        """Compute mutual information from joint distribution."""
        # Compute marginals
        p_x = joint_dist.sum(axis=1)
        p_y = joint_dist.sum(axis=0)
        
        # Compute MI
        mi = 0.0
        for i in range(joint_dist.shape[0]):
            for j in range(joint_dist.shape[1]):
                if joint_dist[i,j] > 0:
                    mi += joint_dist[i,j] * np.log(
                        joint_dist[i,j] / (p_x[i] * p_y[j])
                    )
        return mi
```

## Applications

### Information Gain
```python
class InformationGain:
    """
    Information gain computation.
    
    Theory:
        - [[expected_information_gain]]
        - [[epistemic_value]]
        - [[active_learning]]
    """
    def __init__(self, prior: Distribution):
        self.prior = prior
    
    def compute_expected_gain(self,
                            likelihood: Distribution,
                            action: Action) -> float:
        """Compute expected information gain for an action."""
        # Compute posterior
        posterior = self.update_distribution(self.prior, likelihood)
        
        # Compute expected KL
        expected_kl = self.expected_kl_divergence(
            posterior, self.prior, likelihood
        )
        
        return expected_kl
```

### Channel Capacity
```python
class InformationChannel:
    """
    Information channel analysis.
    
    Theory:
        - [[channel_capacity]]
        - [[rate_distortion]]
        - [[coding_theory]]
    """
    def __init__(self,
                 transition_matrix: np.ndarray,
                 input_dist: np.ndarray = None):
        self.P = transition_matrix
        self.input_dist = input_dist or self._optimize_input()
    
    def compute_capacity(self) -> float:
        """Compute channel capacity."""
        if self.input_dist is None:
            self.input_dist = self._optimize_input()
        
        return self.compute_mutual_information(
            self.input_dist, self.P
        )
```

## Information Geometry

### Fisher Information
```python
class FisherInformation:
    """
    Fisher information computation.
    
    Theory:
        - [[fisher_information_matrix]]
        - [[natural_gradient]]
        - [[cramer_rao_bound]]
    """
    @staticmethod
    def compute_fisher_matrix(distribution: Distribution,
                            params: np.ndarray) -> np.ndarray:
        """Compute Fisher information matrix."""
        # Compute score function
        score = distribution.score_function(params)
        
        # Compute outer product
        fisher = np.outer(score, score)
        
        return np.mean(fisher, axis=0)
```

### Information Manifolds
- [[statistical_manifold]] - Manifold structure
- [[information_geometry]] - Geometric methods
- [[natural_gradient]] - Natural gradients
- [[wasserstein_distance]] - Optimal transport

## Numerical Methods

### Entropy Estimation
```python
class EntropyEstimation:
    """
    Entropy estimation methods.
    
    Theory:
        - [[kernel_density_estimation]]
        - [[nearest_neighbor_estimation]]
        - [[maximum_likelihood_estimation]]
    """
    @staticmethod
    def knn_entropy_estimate(samples: np.ndarray,
                           k: int = 3) -> float:
        """Estimate entropy using k-nearest neighbors."""
        n_samples = len(samples)
        distances = compute_knn_distances(samples, k)
        return (np.log(n_samples) - np.mean(np.log(distances)) + 
                np.log(2) + euler_gamma)
```

### Sampling Methods
- [[importance_sampling]] - IS techniques
- [[monte_carlo_methods]] - MC estimation
- [[sequential_estimation]] - Online methods
- [[adaptive_sampling]] - Adaptive approaches

## Implementation Considerations

### Numerical Stability
- [[log_domain_computation]] - Log-space methods
- [[stable_kl_divergence]] - Stable KL computation
- [[overflow_prevention]] - Overflow handling
- [[underflow_prevention]] - Underflow handling

### Computational Efficiency
- [[vectorized_operations]] - Vectorization
- [[parallel_computation]] - Parallelization
- [[sparse_representations]] - Sparse methods
- [[approximate_methods]] - Approximations

## Validation Framework

### Quality Metrics
```python
class InformationMetrics:
    """Quality metrics for information measures."""
    
    @staticmethod
    def entropy_error(true_entropy: float,
                     estimated_entropy: float) -> float:
        """Compute entropy estimation error."""
        return np.abs(true_entropy - estimated_entropy)
    
    @staticmethod
    def divergence_symmetry(p: np.ndarray,
                          q: np.ndarray,
                          divergence_fn: Callable) -> float:
        """Check divergence symmetry."""
        forward = divergence_fn(p, q)
        reverse = divergence_fn(q, p)
        return np.abs(forward - reverse)
```

### Performance Analysis
- [[estimation_accuracy]] - Accuracy metrics
- [[convergence_rates]] - Convergence analysis
- [[sample_complexity]] - Sample requirements
- [[computational_cost]] - Cost analysis

## Integration Points

### Theory Integration
- [[active_inference]] - Active inference
- [[predictive_coding]] - Predictive processing
- [[variational_inference]] - VI methods
- [[optimal_control]] - Control theory

### Implementation Links
- [[information_metrics]] - Information measures
- [[entropy_estimators]] - Entropy estimation
- [[divergence_computations]] - Divergence computation
- [[sampling_methods]] - Sampling approaches

## Documentation Links
- [[../../docs/research/research_documentation_index|Research Documentation]]
- [[../../docs/guides/implementation_guides_index|Implementation Guides]]
- [[../../docs/api/api_documentation_index|API Documentation]]
- [[../../docs/examples/usage_examples_index|Usage Examples]]

## References
- [[cover_thomas]] - Information Theory
- [[amari_nagaoka]] - Information Geometry
- [[mackay]] - Information Theory and Learning
- [[nielsen_chuang]] - Quantum Information 