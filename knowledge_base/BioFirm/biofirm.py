"""
Main BioFirm implementation.
Integrates all components for bioregional evaluation and stewardship.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import yaml
from pathlib import Path
import logging

from .core.state_spaces import StateSpace, BioregionalState
from .core.observation import (
    ObservationModel, HierarchicalObservation, ObservationAggregator
)
from .core.transition import (
    TransitionModel, HierarchicalTransition, EcologicalConstraints
)
from .core.stewardship import (
    Intervention, StewardshipMetrics, StewardshipMode,
    AdaptiveComanagement, BioregionalStewardship
)
from .visualization.plotting import BioregionalVisualization

logger = logging.getLogger(__name__)

class BioFirm:
    """Main BioFirm class for bioregional active inference."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize BioFirm framework.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_components()
        self._initialize_state()
        self.visualization = BioregionalVisualization()
        
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self._validate_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
            
    def _validate_config(self, config: Dict) -> bool:
        """Validate configuration structure."""
        required_sections = [
            'system',
            'state_spaces',
            'observation',
            'transition',
            'stewardship',
            'visualization'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        return True
        
    def _setup_components(self):
        """Set up all framework components."""
        # Set up state spaces
        self.state_spaces = self._setup_state_spaces()
        
        # Set up observation model
        self.observation_model = self._setup_observation_model()
        
        # Set up transition model
        self.transition_model = self._setup_transition_model()
        
        # Set up stewardship
        self.stewardship = self._setup_stewardship()
        
    def _setup_state_spaces(self) -> Dict[str, StateSpace]:
        """Set up state spaces for each scale."""
        spaces = {}
        for scale, config in self.config['state_spaces'].items():
            spaces[scale] = StateSpace(
                dimensions=config['dimensions'],
                labels=config['labels'],
                mappings=config['mappings'],
                scale=scale
            )
        return spaces
        
    def _setup_observation_model(self) -> HierarchicalObservation:
        """Set up hierarchical observation model."""
        models = {}
        for scale, config in self.config['observation'].items():
            models[scale] = ObservationModel(
                state_space=self.state_spaces[scale],
                observation_space=StateSpace(**config['observation_space']),
                likelihood_matrix=np.array(config['likelihood_matrix']),
                noise_model=config.get('noise_model', 'gaussian'),
                precision=config.get('precision', 1.0)
            )
            
        scale_couplings = {
            tuple(k.split('_to_')): v
            for k, v in self.config['observation']['couplings'].items()
        }
        
        return HierarchicalObservation(models, scale_couplings)
        
    def _setup_transition_model(self) -> HierarchicalTransition:
        """Set up hierarchical transition model."""
        models = {}
        for scale, config in self.config['transition'].items():
            models[scale] = TransitionModel(
                state_space=self.state_spaces[scale],
                action_space=StateSpace(**config['action_space']),
                transition_matrices={
                    action: np.array(matrix)
                    for action, matrix in config['transition_matrices'].items()
                },
                temporal_horizon=config['temporal_horizon'],
                control_modes=config.get('control_modes', ['homeostatic'])
            )
            
        scale_couplings = {
            tuple(k.split('_to_')): v
            for k, v in self.config['transition']['couplings'].items()
        }
        
        return HierarchicalTransition(models, scale_couplings)
        
    def _setup_stewardship(self) -> BioregionalStewardship:
        """Set up stewardship framework."""
        config = self.config['stewardship']
        
        # Create stewardship mode
        mode = AdaptiveComanagement(
            stakeholder_weights=config['stakeholder_weights'],
            learning_rate=config.get('learning_rate', 0.1)
        )
        
        return BioregionalStewardship(
            observation_model=self.observation_model,
            transition_model=self.transition_model,
            stewardship_mode=mode
        )
        
    def _initialize_state(self):
        """Initialize system state."""
        self.current_states = {}
        self.target_states = {}
        
        for scale, config in self.config['system']['initial_states'].items():
            self.current_states[scale] = BioregionalState(**config)
            
        for scale, config in self.config['system']['target_states'].items():
            self.target_states[scale] = BioregionalState(**config)
            
    def evaluate_system(self) -> Dict[str, float]:
        """Evaluate current system state."""
        return self.stewardship.evaluate_system(
            self.current_states,
            self.target_states
        )
        
    def plan_interventions(self,
                         constraints: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, List[Intervention]]:
        """Plan interventions across scales."""
        if constraints is None:
            constraints = self.config['stewardship']['default_constraints']
            
        return self.stewardship.plan_interventions(
            self.current_states,
            constraints
        )
        
    def apply_intervention(self,
                         intervention: Intervention,
                         scale: str) -> BioregionalState:
        """Apply intervention and update state."""
        # Get current state
        current_state = self.current_states[scale]
        
        # Predict next state
        next_state = self.transition_model.models[scale].predict_next_state(
            current_state,
            intervention.type
        )
        
        # Apply ecological constraints
        constraints = EcologicalConstraints(
            self.config['system']['ecological_constraints']
        )
        constrained_state = constraints.apply_constraints(next_state)
        
        # Update current state
        self.current_states[scale] = constrained_state
        
        # Update metrics
        self.stewardship.update_metrics(
            self.current_states,
            {scale: [intervention]}
        )
        
        return constrained_state
        
    def visualize_state(self,
                       time_series: Optional[np.ndarray] = None,
                       scale: Optional[str] = None):
        """Visualize system state."""
        if scale is None:
            scale = list(self.current_states.keys())[0]
            
        return self.visualization.plot_system_state(
            self.current_states[scale],
            time_series
        )
        
    def visualize_intervention(self,
                             before_state: BioregionalState,
                             after_state: BioregionalState,
                             intervention: Intervention):
        """Visualize intervention impacts."""
        return self.visualization.plot_intervention_impacts(
            before_state,
            after_state,
            intervention.__dict__
        )
        
    def visualize_cross_scale(self,
                            states: Optional[Dict[str, np.ndarray]] = None):
        """Visualize cross-scale dynamics."""
        if states is None:
            states = {
                scale: np.array([state.to_vector()])
                for scale, state in self.current_states.items()
            }
            
        scales = list(states.keys())
        interactions = np.array([
            [self.transition_model.scale_couplings.get(
                (scale1, scale2), 0.0
            ) for scale2 in scales]
            for scale1 in scales
        ])
        
        return self.visualization.plot_cross_scale_dynamics(
            states, scales, interactions
        ) 