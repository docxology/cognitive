"""
Nestmate Agent Implementation

This module implements the Nestmate agent class, which represents an individual ant
in the colony using the Free Energy Principle (FEP) and Active Inference framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    """Possible task types for a Nestmate agent."""
    FORAGING = "foraging"
    MAINTENANCE = "maintenance"
    NURSING = "nursing"
    DEFENSE = "defense"
    EXPLORATION = "exploration"

@dataclass
class Observation:
    """Container for sensory observations."""
    pheromone: np.ndarray  # Pheromone gradients
    food: np.ndarray       # Food source locations
    nestmates: np.ndarray  # Other agent positions
    obstacles: np.ndarray  # Obstacle positions
    nest: np.ndarray      # Nest location/gradient

class GenerativeModel(nn.Module):
    """Hierarchical generative model for active inference."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Model dimensions
        self.obs_dim = config['dimensions']['observations']
        self.state_dim = config['dimensions']['states']
        self.action_dim = config['dimensions']['actions']
        self.temporal_horizon = config['dimensions']['planning_horizon']
        
        # Hierarchical layers
        self.layers = nn.ModuleList([
            nn.Linear(self.state_dim, self.state_dim) 
            for _ in range(config['active_inference']['model']['hierarchical_levels'])
        ])
        
        # State transition model (dynamics)
        self.transition = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.state_dim * 2),
            nn.ReLU(),
            nn.Linear(self.state_dim * 2, self.state_dim)
        )
        
        # Observation model
        self.observation = nn.Sequential(
            nn.Linear(self.state_dim, self.obs_dim * 2),
            nn.ReLU(),
            nn.Linear(self.obs_dim * 2, self.obs_dim)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(self.state_dim, self.action_dim * 2),
            nn.ReLU(),
            nn.Linear(self.action_dim * 2, self.action_dim)
        )
        
        # Precision parameters
        self.alpha = nn.Parameter(torch.ones(1))  # Precision of beliefs
        self.beta = nn.Parameter(torch.ones(1))   # Precision of policies
        
    def forward(self, 
                state: torch.Tensor, 
                action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the generative model."""
        
        # Hierarchical state processing
        for layer in self.layers:
            state = F.relu(layer(state))
            
        # Generate observations
        predicted_obs = self.observation(state)
        
        # If action provided, predict next state
        if action is not None:
            state_action = torch.cat([state, action], dim=-1)
            next_state = self.transition(state_action)
            return predicted_obs, next_state
            
        return predicted_obs, None
    
    def infer_state(self, 
                   obs: torch.Tensor, 
                   prev_state: Optional[torch.Tensor] = None,
                   n_steps: int = 10) -> torch.Tensor:
        """Infer hidden state through iterative message passing."""
        
        if prev_state is None:
            state = torch.zeros(obs.shape[0], self.state_dim)
        else:
            state = prev_state
            
        state.requires_grad = True
        optimizer = torch.optim.Adam([state], lr=0.1)
        
        for _ in range(n_steps):
            optimizer.zero_grad()
            
            # Prediction errors
            pred_obs, _ = self.forward(state)
            obs_error = F.mse_loss(pred_obs, obs)
            
            if prev_state is not None:
                state_error = F.mse_loss(state, prev_state)
                loss = obs_error + self.alpha * state_error
            else:
                loss = obs_error
                
            loss.backward()
            optimizer.step()
            
        return state.detach()
    
    def select_action(self, 
                     state: torch.Tensor, 
                     temperature: float = 1.0) -> torch.Tensor:
        """Select action using active inference."""
        
        # Get action distribution
        action_logits = self.policy(state)
        action_probs = F.softmax(action_logits / temperature, dim=-1)
        
        # Sample action
        action = torch.multinomial(action_probs, 1)
        
        return action

class Nestmate:
    """
    Individual ant agent implementing active inference for decision making.
    """
    
    def __init__(self, config: dict):
        """Initialize Nestmate agent."""
        self.config = config
        
        # Physical state
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.orientation = 0.0
        self.energy = config['physical']['energy']['initial']
        
        # Task state
        self.current_task = TaskType.EXPLORATION
        self.carrying = None
        
        # Sensory state
        self.observations = Observation(
            pheromone=np.zeros(config['sensors']['pheromone']['types'].__len__()),
            food=np.zeros(2),
            nestmates=np.zeros(2),
            obstacles=np.zeros(2),
            nest=np.zeros(2)
        )
        
        # Active inference components
        self.generative_model = GenerativeModel(config)
        self.current_state = None
        self.previous_action = None
        
        # Memory
        self.memory = {
            'spatial': [],
            'temporal': [],
            'social': []
        }
        
        # Learning parameters
        self.learning_rate = config['learning']['parameters']['learning_rate']
        self.exploration_rate = config['learning']['parameters']['exploration_rate']
        
    def update(self, observation: Observation) -> np.ndarray:
        """
        Update agent state and select action using active inference.
        
        Args:
            observation: Current sensory observations
            
        Returns:
            action: Selected action as numpy array
        """
        # Convert observation to tensor
        obs_tensor = torch.tensor(self._preprocess_observation(observation))
        
        # State inference
        inferred_state = self.generative_model.infer_state(
            obs_tensor, 
            prev_state=self.current_state
        )
        self.current_state = inferred_state
        
        # Action selection
        action = self.generative_model.select_action(
            inferred_state,
            temperature=self.config['active_inference']['free_energy']['temperature']
        )
        
        # Update memory
        self._update_memory(observation, action)
        
        # Update internal state
        self._update_internal_state()
        
        return action.numpy()
    
    def _preprocess_observation(self, observation: Observation) -> np.ndarray:
        """Preprocess raw observations into model input format."""
        # Combine all observations into single vector
        obs_vector = np.concatenate([
            observation.pheromone,
            observation.food,
            observation.nestmates,
            observation.obstacles,
            observation.nest
        ])
        
        # Normalize
        obs_vector = (obs_vector - obs_vector.mean()) / (obs_vector.std() + 1e-8)
        
        return obs_vector
    
    def _update_memory(self, observation: Observation, action: torch.Tensor):
        """Update agent's memory systems."""
        # Spatial memory
        self.memory['spatial'].append({
            'position': self.position.copy(),
            'observation': observation,
            'timestamp': None  # Add actual timestamp in implementation
        })
        
        # Temporal memory
        self.memory['temporal'].append({
            'state': self.current_state.detach().numpy(),
            'action': action.numpy(),
            'reward': self._compute_reward(observation)
        })
        
        # Social memory (interactions with other agents)
        if np.any(observation.nestmates):
            self.memory['social'].append({
                'nestmate_positions': observation.nestmates.copy(),
                'interaction_type': self._classify_interaction(observation)
            })
            
        # Maintain memory size limits
        for memory_type in self.memory:
            if len(self.memory[memory_type]) > self.config['memory'][memory_type]['capacity']:
                self.memory[memory_type].pop(0)
                
    def _update_internal_state(self):
        """Update agent's internal state variables."""
        # Update energy
        self.energy -= self.config['physical']['energy']['consumption_rate']
        if self.carrying is not None:
            self.energy -= self.config['physical']['energy']['consumption_rate'] * 2
            
        # Update task if needed
        if self._should_switch_task():
            self._switch_task()
            
        # Update learning parameters
        self.exploration_rate *= self.config['learning']['parameters']['decay_rate']
        self.exploration_rate = max(
            self.exploration_rate,
            self.config['learning']['parameters']['min_exploration']
        )
        
    def _compute_reward(self, observation: Observation) -> float:
        """Compute reward signal from current observation."""
        reward = 0.0
        
        # Task-specific rewards
        if self.current_task == TaskType.FORAGING:
            reward += np.sum(observation.food) * self.config['active_inference']['preferences']['food_weight']
            
        # Distance to nest reward
        nest_distance = np.linalg.norm(observation.nest)
        reward -= nest_distance * self.config['active_inference']['preferences']['home_weight']
        
        # Safety reward (avoiding obstacles)
        obstacle_penalty = np.sum(1.0 / (1.0 + np.linalg.norm(observation.obstacles, axis=1)))
        reward -= obstacle_penalty * self.config['active_inference']['preferences']['safety_weight']
        
        # Social reward
        if np.any(observation.nestmates):
            social_reward = self.config['active_inference']['preferences']['social_weight']
            reward += social_reward
            
        return reward
    
    def _should_switch_task(self) -> bool:
        """Determine if agent should switch its current task."""
        # Energy-based switching
        if self.energy < self.config['physical']['energy']['critical_level']:
            return True
            
        # Random switching based on flexibility
        if np.random.random() < self.config['behavior']['task_switching']['flexibility']:
            return True
            
        return False
    
    def _switch_task(self):
        """Switch to a new task based on current conditions."""
        # Get valid task options
        valid_tasks = list(TaskType)
        if self.current_task in valid_tasks:
            valid_tasks.remove(self.current_task)
            
        # Select new task (can be made more sophisticated)
        self.current_task = np.random.choice(valid_tasks)
        
    def _classify_interaction(self, observation: Observation) -> str:
        """Classify type of interaction with nearby nestmates."""
        # Simple distance-based classification
        distances = np.linalg.norm(observation.nestmates, axis=1)
        if np.any(distances < 1.0):
            return "direct"
        elif np.any(distances < 3.0):
            return "indirect"
        return "none" 