"""
Active Inference method dispatcher and high-level abstractions.
Provides a clean interface for dispatching active inference operations
to appropriate low-level implementations.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from .base import ActiveInferenceModel, ModelState
from ..matrices.matrix_ops import MatrixOps, MatrixInitializer

class InferenceMethod(Enum):
    """Supported inference methods."""
    VARIATIONAL = "variational"
    SAMPLING = "sampling"
    MEAN_FIELD = "mean_field"
    
class PolicyType(Enum):
    """Supported policy types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HIERARCHICAL = "hierarchical"

@dataclass
class InferenceConfig:
    """Configuration for inference method dispatch."""
    method: InferenceMethod
    policy_type: PolicyType
    temporal_horizon: int
    learning_rate: float
    precision_init: float
    use_gpu: bool = False
    custom_params: Optional[Dict[str, Any]] = None

class ActiveInferenceDispatcher:
    """
    Dispatcher for Active Inference operations.
    Provides high-level interface and handles routing to specific implementations.
    """
    
    def __init__(self, config: InferenceConfig):
        """Initialize dispatcher with configuration."""
        self.config = config
        self._setup_implementations()
        self._initialize_matrices()
        
    def _setup_implementations(self):
        """Set up mapping of operations to implementations."""
        self._implementations = {
            InferenceMethod.VARIATIONAL: {
                'belief_update': self._variational_belief_update,
                'policy_inference': self._variational_policy_inference
            },
            InferenceMethod.SAMPLING: {
                'belief_update': self._sampling_belief_update,
                'policy_inference': self._sampling_policy_inference
            },
            InferenceMethod.MEAN_FIELD: {
                'belief_update': self._mean_field_belief_update,
                'policy_inference': self._mean_field_policy_inference
            }
        }
        
    def _initialize_matrices(self):
        """Initialize required matrices based on configuration."""
        self.matrix_ops = MatrixOps()
        self.matrix_init = MatrixInitializer()
        
    def dispatch_belief_update(self, 
                             observation: np.ndarray,
                             current_state: ModelState,
                             **kwargs) -> np.ndarray:
        """
        Dispatch belief update to appropriate implementation.
        
        Args:
            observation: Current observation
            current_state: Current model state
            **kwargs: Additional parameters for specific implementations
            
        Returns:
            Updated beliefs
        """
        update_fn = self._implementations[self.config.method]['belief_update']
        return update_fn(observation, current_state, **kwargs)
        
    def dispatch_policy_inference(self,
                                state: ModelState,
                                goal_prior: Optional[np.ndarray] = None,
                                **kwargs) -> np.ndarray:
        """
        Dispatch policy inference to appropriate implementation.
        
        Args:
            state: Current model state
            goal_prior: Optional prior over goal states
            **kwargs: Additional parameters for specific implementations
            
        Returns:
            Inferred policy distributions
        """
        inference_fn = self._implementations[self.config.method]['policy_inference']
        return inference_fn(state, goal_prior, **kwargs)
    
    def _variational_belief_update(self,
                                 observation: np.ndarray,
                                 state: ModelState,
                                 **kwargs) -> np.ndarray:
        """Variational implementation of belief updates."""
        # Implementation details for variational belief updates
        prediction = np.dot(state.beliefs, kwargs.get('generative_matrix', np.eye(len(state.beliefs))))
        prediction_error = observation - prediction
        belief_update = state.precision * prediction_error
        return state.beliefs + belief_update
        
    def _sampling_belief_update(self,
                              observation: np.ndarray,
                              state: ModelState,
                              **kwargs) -> np.ndarray:
        """Sampling-based implementation of belief updates."""
        # Implementation for sampling-based updates
        raise NotImplementedError("Sampling-based belief updates not yet implemented")
        
    def _mean_field_belief_update(self,
                                observation: np.ndarray,
                                state: ModelState,
                                **kwargs) -> np.ndarray:
        """Mean-field implementation of belief updates."""
        # Implementation for mean-field updates
        raise NotImplementedError("Mean-field belief updates not yet implemented")
        
    def _variational_policy_inference(self,
                                    state: ModelState,
                                    goal_prior: Optional[np.ndarray] = None,
                                    **kwargs) -> np.ndarray:
        """Variational implementation of policy inference."""
        # Implementation for variational policy inference
        if goal_prior is None:
            goal_prior = np.ones(len(state.policies)) / len(state.policies)
        
        expected_free_energy = self._calculate_expected_free_energy(
            state, goal_prior, **kwargs)
        return self.matrix_ops.softmax(-expected_free_energy)
        
    def _sampling_policy_inference(self,
                                 state: ModelState,
                                 goal_prior: Optional[np.ndarray] = None,
                                 **kwargs) -> np.ndarray:
        """Sampling-based implementation of policy inference."""
        raise NotImplementedError("Sampling-based policy inference not yet implemented")
        
    def _mean_field_policy_inference(self,
                                   state: ModelState,
                                   goal_prior: Optional[np.ndarray] = None,
                                   **kwargs) -> np.ndarray:
        """Mean-field implementation of policy inference."""
        raise NotImplementedError("Mean-field policy inference not yet implemented")
        
    def _calculate_expected_free_energy(self,
                                      state: ModelState,
                                      goal_prior: np.ndarray,
                                      **kwargs) -> np.ndarray:
        """Calculate expected free energy for policy evaluation."""
        # Basic implementation - can be extended based on specific needs
        pragmatic_value = -np.log(goal_prior + 1e-8)  # Avoid log(0)
        epistemic_value = self._calculate_epistemic_value(state)
        return pragmatic_value + epistemic_value
        
    def _calculate_epistemic_value(self, state: ModelState) -> np.ndarray:
        """Calculate epistemic value component of expected free energy."""
        # Simple implementation - can be extended
        return -state.prediction_error * np.ones(len(state.policies))

class ActiveInferenceFactory:
    """Factory for creating Active Inference instances with specific configurations."""
    
    @staticmethod
    def create(config: InferenceConfig) -> ActiveInferenceDispatcher:
        """Create an Active Inference dispatcher with specified configuration."""
        return ActiveInferenceDispatcher(config)
    
    @staticmethod
    def create_from_yaml(config_path: Union[str, Path]) -> ActiveInferenceDispatcher:
        """Create an Active Inference dispatcher from YAML configuration."""
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = InferenceConfig(
            method=InferenceMethod(config_dict['method']),
            policy_type=PolicyType(config_dict['policy_type']),
            temporal_horizon=config_dict['temporal_horizon'],
            learning_rate=config_dict['learning_rate'],
            precision_init=config_dict['precision_init'],
            use_gpu=config_dict.get('use_gpu', False),
            custom_params=config_dict.get('custom_params', None)
        )
        return ActiveInferenceFactory.create(config) 