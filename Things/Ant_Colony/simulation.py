"""
Simulation Runner for Ant Colony Simulation

This module provides the main simulation runner that coordinates the colony,
environment, and visualization components.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import time
import logging
import yaml
import h5py
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from .visualization import ColonyVisualizer
from .colony import Colony
from .environment import World
from .utils.data_collection import DataCollector
from agents.nestmate import TaskType

class Simulation:
    """Main simulation runner class."""
    
    def __init__(self, config_path: str):
        """Initialize simulation with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Initialize components
        self.environment = World(self.config['environment'])
        self.colony = Colony(self.config['colony'], self.environment)
        self.visualizer = ColonyVisualizer(self.config['visualization'])
        self.data_collector = DataCollector(self.config['data_collection'])
        
        # Simulation state
        self.current_step = 0
        self.max_steps = self.config['simulation']['max_steps']
        self.timestep = self.config['simulation']['timestep']
        self.paused = False
        self.data = {
            'time': [],
            'resources': [],
            'task_distribution': [],
            'efficiency_metrics': []
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config['debug']['logging']
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_config['file']
        )
        self.logger = logging.getLogger(__name__)
        
    def step(self):
        """Execute one simulation step."""
        # Update environment
        self.environment.update(self.timestep)
        
        # Update colony
        self.colony.update(self.timestep)
        
        # Collect data
        self._collect_data()
        
        # Increment step counter
        self.current_step += 1
        
    def _collect_data(self):
        """Collect simulation data for analysis and visualization."""
        self.data['time'].append(self.current_step * self.timestep)
        
        # Collect resource data
        resource_data = {
            rtype: sum(r.amount for r in resources)
            for rtype, resources in self.environment.resources.items()
        }
        self.data['resources'].append(resource_data)
        
        # Collect task distribution
        task_dist = self.colony.get_task_distribution()
        self.data['task_distribution'].append(task_dist)
        
        # Collect efficiency metrics
        metrics = self.colony.compute_efficiency_metrics()
        self.data['efficiency_metrics'].append(metrics)
        
    def run(self, headless: bool = False):
        """Run the simulation."""
        self.logger.info("Starting simulation...")
        
        if not headless:
            # Create and show animation
            animation = self.visualizer.create_animation(self)
            plt.show()
        else:
            # Run without visualization
            try:
                while self.current_step < self.max_steps and not self.paused:
                    self.step()
                    
                    # Save data periodically
                    if self.current_step % self.config['data_collection']['frequency'] == 0:
                        self.data_collector.save_data(self.data)
                        
            except KeyboardInterrupt:
                self.logger.info("Simulation interrupted by user")
            finally:
                # Save final data
                self.data_collector.save_data(self.data)
                
        self.logger.info(f"Simulation completed after {self.current_step} steps")
        
    def pause(self):
        """Pause the simulation."""
        self.paused = True
        self.logger.info("Simulation paused")
        
    def resume(self):
        """Resume the simulation."""
        self.paused = False
        self.logger.info("Simulation resumed")
        
    def reset(self):
        """Reset the simulation to initial state."""
        self.logger.info("Resetting simulation...")
        self.current_step = 0
        self.environment.reset()
        self.colony.reset()
        self.data = {
            'time': [],
            'resources': [],
            'task_distribution': [],
            'efficiency_metrics': []
        }
        
    def _update_visualization(self):
        """Update visualization plots."""
        # Clear all axes
        for ax in [self.ax_main, self.ax_resources, self.ax_tasks, self.ax_metrics]:
            ax.clear()
            
        # Main simulation view
        self._plot_simulation_state()
        
        # Resource levels
        self._plot_resource_levels()
        
        # Task distribution
        self._plot_task_distribution()
        
        # Efficiency metrics
        self._plot_efficiency_metrics()
        
        plt.draw()
        plt.pause(0.01)
        
    def _plot_simulation_state(self):
        """Plot current simulation state."""
        # Plot terrain
        terrain = self.environment.terrain.height_map
        self.ax_main.imshow(terrain, cmap='terrain')
        
        # Plot agents
        agent_positions = np.array([agent.position for agent in self.colony.agents])
        agent_colors = [self._get_task_color(agent.current_task) for agent in self.colony.agents]
        
        if len(agent_positions) > 0:
            self.ax_main.scatter(agent_positions[:, 0], agent_positions[:, 1], 
                               c=agent_colors, alpha=0.6)
            
        # Plot resources
        resource_positions = np.array([[r.position.x, r.position.y] for r in self.environment.resources])
        if len(resource_positions) > 0:
            self.ax_main.scatter(resource_positions[:, 0], resource_positions[:, 1], 
                               c='yellow', marker='*')
            
        # Plot nest
        self.ax_main.scatter([self.colony.nest_position.x], [self.colony.nest_position.y],
                           c='white', marker='s', s=100)
                           
        self.ax_main.set_title(f'Step {self.current_step}')
        
    def _plot_resource_levels(self):
        """Plot resource level history."""
        if len(self.data['time']) > 0:
            for resource_type in self.colony.resources.keys():
                values = [d[resource_type] for d in self.data['resources']]
                self.ax_resources.plot(self.data['time'], values, label=resource_type)
                
            self.ax_resources.legend()
            self.ax_resources.set_xlabel('Time')
            self.ax_resources.set_ylabel('Amount')
            
    def _plot_task_distribution(self):
        """Plot task distribution."""
        if len(self.data['task_distribution']) > 0:
            latest_dist = self.data['task_distribution'][-1]
            tasks = list(latest_dist.keys())
            counts = [latest_dist[task] for task in tasks]
            
            colors = [self._get_task_color(task) for task in tasks]
            self.ax_tasks.bar(range(len(tasks)), counts, color=colors)
            self.ax_tasks.set_xticks(range(len(tasks)))
            self.ax_tasks.set_xticklabels([task.value for task in tasks], rotation=45)
            
    def _plot_efficiency_metrics(self):
        """Plot efficiency metrics."""
        if len(self.data['efficiency_metrics']) > 0:
            latest_metrics = self.data['efficiency_metrics'][-1]
            metrics = list(latest_metrics.keys())
            values = [latest_metrics[metric] for metric in metrics]
            
            self.ax_metrics.bar(range(len(metrics)), values)
            self.ax_metrics.set_xticks(range(len(metrics)))
            self.ax_metrics.set_xticklabels(metrics, rotation=45)
            self.ax_metrics.set_ylim(0, 1)
            
    def _get_task_color(self, task: TaskType) -> str:
        """Get color for visualization based on task type."""
        color_map = {
            TaskType.FORAGING: 'green',
            TaskType.MAINTENANCE: 'blue',
            TaskType.NURSING: 'purple',
            TaskType.DEFENSE: 'red',
            TaskType.EXPLORATION: 'orange'
        }
        return color_map.get(task, 'gray')
        
    def _save_data(self):
        """Save simulation data to file."""
        if not self.config['data']['export']['enabled']:
            return
            
        # Create output directory if it doesn't exist
        output_dir = Path(self.config['data']['storage']['path'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to HDF5
        if 'hdf5' in self.config['data']['storage']['format']:
            self._save_to_hdf5(output_dir / f'simulation_data_{self.current_step}.h5')
            
        # Save to CSV
        if 'csv' in self.config['data']['export']['format']:
            self._save_to_csv(output_dir)
            
    def _save_to_hdf5(self, filepath: Path):
        """Save data to HDF5 format."""
        with h5py.File(filepath, 'w') as f:
            # Create groups
            sim_group = f.create_group('simulation')
            colony_group = f.create_group('colony')
            perf_group = f.create_group('performance')
            
            # Save simulation data
            sim_group.create_dataset('time', data=np.array(self.data['time']))
            sim_group.create_dataset('step', data=self.current_step)
            
            # Save colony data
            colony_group.create_dataset('population', data=np.array(self.colony.stats.population))
            
            # Create datasets for dictionary data
            for key in ['resources', 'task_distribution', 'efficiency_metrics']:
                if len(self.data[key]) > 0:
                    group = colony_group.create_group(key)
                    for metric_key in self.data[key][0].keys():
                        values = [d[metric_key] for d in self.data[key]]
                        group.create_dataset(metric_key, data=np.array(values))
                        
            # Save performance data
            perf_group.create_dataset('step_times', data=np.array(self.performance_metrics['step_times']))
            
    def _save_to_csv(self, output_dir: Path):
        """Save data to CSV format."""
        # Save basic metrics
        pd.DataFrame({
            'time': self.data['time'],
            'population': self.colony.stats.population
        }).to_csv(output_dir / 'basic_metrics.csv', index=False)
        
        # Save resource data
        if len(self.data['resources']) > 0:
            pd.DataFrame(self.data['resources']).to_csv(
                output_dir / 'resources.csv', index=False
            )
            
        # Save task distribution
        if len(self.data['task_distribution']) > 0:
            pd.DataFrame(self.data['task_distribution']).to_csv(
                output_dir / 'task_distribution.csv', index=False
            )
            
        # Save efficiency metrics
        if len(self.data['efficiency_metrics']) > 0:
            pd.DataFrame(self.data['efficiency_metrics']).to_csv(
                output_dir / 'efficiency_metrics.csv', index=False
            )
            
    def _monitor_performance(self):
        """Monitor and log performance metrics."""
        if len(self.performance_metrics['step_times']) > 0:
            avg_step_time = np.mean(self.performance_metrics['step_times'][-100:])
            current_fps = 1.0 / avg_step_time
            
            self.performance_metrics['fps'].append(current_fps)
            
            self.logger.info(f"Current FPS: {current_fps:.2f}")
            
            # Check performance against targets
            if current_fps < self.config['performance']['optimization']['target_fps']:
                self._optimize_performance()
                
    def _optimize_performance(self):
        """Attempt to optimize simulation performance."""
        if not self.config['performance']['optimization']['auto_tune']:
            return
            
        # Example optimization: Reduce visualization frequency
        if self.config['visualization']['realtime']['enabled']:
            current_freq = self.config['visualization']['realtime']['update_frequency']
            self.config['visualization']['realtime']['update_frequency'] = min(current_freq * 2, 100)
            
        self.logger.info("Adjusted visualization frequency for performance optimization")
        
    def _cleanup(self):
        """Cleanup resources and save final state."""
        # Save final data
        self._save_data()
        
        # Close visualization
        if self.config['visualization']['realtime']['enabled']:
            plt.close(self.fig)
            
        # Log final statistics
        self.logger.info(f"Simulation completed after {self.current_step} steps")
        self.logger.info(f"Final population: {self.colony.stats.population}")
        self.logger.info(f"Final resource levels: {self.colony.stats.resource_levels}")
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Ant Colony Simulation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--headless', action='store_true',
                       help='Run without visualization')
    args = parser.parse_args()
    
    # Create and run simulation
    sim = Simulation(args.config)
    sim.run(headless=args.headless) 