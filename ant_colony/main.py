"""
Main entry point for the ant colony simulation.
"""

import argparse
import yaml
import numpy as np
from ant_colony.visualization.renderer import SimulationRenderer
from ant_colony.agents.nestmate import Nestmate, Position, TaskType
from dataclasses import dataclass
from typing import List

@dataclass
class FoodSource:
    """Represents a food source in the environment."""
    position: Position
    size: float
    value: float
    remaining: float

@dataclass
class Obstacle:
    """Represents an obstacle in the environment."""
    position: Position
    size: float

class Simulation:
    """Main simulation class."""
    
    def __init__(self, config_path: str):
        """Initialize simulation with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set random seed
        np.random.seed(self.config['simulation']['random_seed'])
        
        # Initialize environment
        self.env_size = self.config['environment']['size']
        self.nest_location = self.config['environment']['nest_location']
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Initialize resources
        self.food_sources = self._create_food_sources()
        self.obstacles = self._create_obstacles()
        
        # Initialize pheromone grids
        self.pheromones = {
            'food': np.zeros(self.env_size),
            'home': np.zeros(self.env_size)
        }
        
        # Setup visualization if enabled
        if self.config['visualization']['realtime']['enabled']:
            self.renderer = SimulationRenderer(self.config)
        else:
            self.renderer = None 