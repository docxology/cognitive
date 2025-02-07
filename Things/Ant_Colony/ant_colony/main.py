"""
Main entry point for the ant colony simulation.
"""

import argparse
import yaml
import numpy as np
from visualization.renderer import SimulationRenderer
from agents.nestmate import Nestmate, Position
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
        if self.config['visualization']['enabled']:
            self.renderer = SimulationRenderer(self.config)
        else:
            self.renderer = None
            
    def _create_agents(self) -> List[Nestmate]:
        """Create initial population of agents."""
        agents = []
        for _ in range(self.config['colony']['initial_population']):
            # Start agents near nest
            x = self.nest_location[0] + np.random.normal(0, 2)
            y = self.nest_location[1] + np.random.normal(0, 2)
            theta = np.random.uniform(0, 2 * np.pi)
            
            agent = Nestmate(self.config['agent'])
            agent.position = Position(x, y, theta)
            agents.append(agent)
            
        return agents
        
    def _create_food_sources(self) -> List[FoodSource]:
        """Create initial food sources."""
        sources = []
        for _ in range(self.config['environment']['food_sources']['count']):
            x = np.random.uniform(0, self.env_size[0])
            y = np.random.uniform(0, self.env_size[1])
            size = np.random.uniform(*self.config['environment']['food_sources']['size_range'])
            value = np.random.uniform(*self.config['environment']['food_sources']['value_range'])
            
            source = FoodSource(
                position=Position(x, y, 0),
                size=size,
                value=value,
                remaining=value
            )
            sources.append(source)
            
        return sources
        
    def _create_obstacles(self) -> List[Obstacle]:
        """Create initial obstacles."""
        obstacles = []
        for _ in range(self.config['environment']['obstacles']['count']):
            x = np.random.uniform(0, self.env_size[0])
            y = np.random.uniform(0, self.env_size[1])
            size = np.random.uniform(*self.config['environment']['obstacles']['size_range'])
            
            obstacle = Obstacle(
                position=Position(x, y, 0),
                size=size
            )
            obstacles.append(obstacle)
            
        return obstacles
        
    def update(self) -> None:
        """Update simulation state for one timestep."""
        dt = self.config['simulation']['timestep']
        
        # Update agents
        world_state = {
            'agents': self.agents,
            'resources': self.food_sources,
            'obstacles': self.obstacles,
            'pheromones': self.pheromones
        }
        
        for agent in self.agents:
            agent.update(dt, world_state)
            
        # Update pheromones
        decay = self.config['environment']['pheromone_decay']
        self.pheromones['food'] *= decay
        self.pheromones['home'] *= decay
        
        # Update visualization
        if self.renderer and self.config['visualization']['enabled']:
            self.renderer.update(world_state)
            
    def run(self) -> None:
        """Run the simulation."""
        max_steps = self.config['simulation']['max_steps']
        
        try:
            for step in range(max_steps):
                self.update()
                
                if step % 100 == 0:
                    print(f"Step {step}/{max_steps}")
                    
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            
        finally:
            if self.renderer:
                self.renderer.show()
                
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ant Colony Simulation")
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    
    simulation = Simulation(args.config)
    simulation.run()
    
if __name__ == '__main__':
    main() 