# Cognitive Ecosystem Modeling Framework

A comprehensive framework for modeling cognitive ecosystems using [[active_inference|Active Inference]], integrated with [[obsidian_linking|Obsidian]] for knowledge management.

## Overview

This project combines cognitive modeling with knowledge management to create a powerful framework for:
- Modeling agent behaviors using [[active_inference|Active Inference]]
- Managing complex [[knowledge_organization|knowledge structures]]
- Visualizing and analyzing [[cognitive_phenomena|cognitive networks]]
- Simulating multi-agent interactions

## Project Structure

See [[ai_folder_structure]] for comprehensive directory organization.

```
📁 templates/               # Template definitions
├── node_templates/        # Base templates for cognitive nodes
│   ├── agent_template.md  # See [[ai_concept_template]]
│   ├── belief_template.md
│   └── ...
│
📁 knowledge_base/         # Knowledge structure
├── cognitive/            # Core cognitive concepts
├── agents/              # Agent definitions
├── beliefs/             # Belief networks
└── ...

📁 src/                    # Source code
├── models/              # Core modeling components
├── utils/              # Utility functions
└── analysis/          # Analysis tools

📁 docs/                   # Documentation (See [[documentation_standards]])
📁 tests/                  # Test suite (See [[testing_guide]])
📁 data/                   # Data storage
```

## Features

### Knowledge Management
- [[obsidian_linking|Obsidian-compatible markdown files]]
- [[linking_completeness|Bidirectional linking]]
- [[ai_concept_template|Template-based node creation]]
- [[ai_validation_framework|Automated relationship tracking]]

### Cognitive Modeling
- [[active_inference|Active Inference implementation]]
- [[belief_updating|Belief updating mechanisms]]
- [[action_selection|Policy selection algorithms]]
- [[predictive_processing|State estimation tools]]

### Analysis & Visualization
- [[network_analysis|Network analysis]]
- [[quality_metrics|Performance metrics]]
- [[visualization_tools|Interactive visualizations]]
- [[simulation_studies|Simulation frameworks]]

## Knowledge Integration Architecture

### Bidirectional Knowledge Graph
The framework leverages [[obsidian_linking|Obsidian's linking capabilities]] to create a living knowledge graph that:
- Enforces [[validation_framework|mathematical and theoretical consistency]]
- Enables [[ai_validation_framework|automated validation]] of relationships
- Supports [[machine_readability|dynamic discovery]] of dependencies
- Facilitates [[research_education|learning through exploration]]

#### Link Types and Semantics
See [[linking_completeness]] for comprehensive linking patterns.

1. Theoretical Dependencies
   ```markdown
   [[measure_theory]] → [[probability_theory]] → [[stochastic_processes]]
   ```
   - Enforces prerequisite knowledge
   - Validates theoretical foundations
   - Ensures consistent notation

2. Implementation Dependencies
   ```markdown
   [[active_inference]] → [[belief_updating]] → [[action_selection]]
   ```
   - Tracks computational requirements
   - Maintains implementation consistency
   - Documents design decisions

3. Validation Links
   ```markdown
   [[testing_guide]] → [[validation_framework]] → [[quality_metrics]]
   ```
   - Ensures rigorous testing
   - Maintains quality standards
   - Documents validation procedures

### Probabilistic Programming Integration

#### Graph Structure Mapping
The repository's link structure directly maps to probabilistic graphical models:
```python
# Example: Converting knowledge links to Bayesian Graph
def build_bayesian_graph(knowledge_base: Path) -> BayesianNetwork:
    """Convert knowledge base links to Bayesian Network.
    See [[ai_semantic_processing]] for details.
    """
    graph = BayesianNetwork()
    
    # Extract links and dependencies
    for file in knowledge_base.glob('**/*.md'):
        links = extract_links(file)
        nodes = create_nodes(links)
        edges = create_edges(links)
        
        # Add to graph with conditional probabilities
        graph.add_nodes(nodes)
        graph.add_edges(edges)
    
    return graph
```

#### Implementation Patterns
See [[package_documentation]] for detailed implementation guidelines.

1. Direct Specification
   ```yaml
   # In matrix_specification.md
   matrix:
     type: observation
     dimensions: [num_states, num_observations]
     distribution: categorical
     parameters:
       prior: dirichlet
       concentration: [1.0, ..., 1.0]
   ```

2. Probabilistic Annotations
   ```python
   @probabilistic_model
   class ObservationModel:
      """Implementation with probabilistic annotations.
      See [[predictive_processing]] for theoretical background.
      """
      def __init__(self):
          self.A = PyroMatrix(dims=['states', 'obs'])
          
      def forward(self, state):
          return pyro.sample('obs', 
                           dist.Categorical(self.A[state]))
   ```

3. Inference Specifications
   ```yaml
   inference:
     method: variational
     guide: mean_field
     optimizer: adam
     parameters:
       learning_rate: 0.01
       num_particles: 10
   ```

### Knowledge Base Integration

#### Automated Validation
```python
def validate_knowledge_base():
    """Validate theoretical consistency.
    See [[ai_validation_framework]] for details.
    """
    # Check link consistency
    validate_theoretical_dependencies()
    
    # Verify probabilistic specifications
    validate_probability_constraints()
    
    # Test implementation coherence
    validate_implementation_patterns()
```

#### Learning Pathways
See [[research_education]] for comprehensive learning integration.

- **Theory**: Follow theoretical dependency chains
  ```markdown
  [[measure_theory]] → [[probability_theory]] → [[active_inference]]
  ```

- **Implementation**: Track implementation requirements
  ```markdown
  [[matrix_design]] → [[numerical_methods]] → [[optimization]]
  ```

- **Validation**: Ensure testing coverage
  ```markdown
  [[unit_tests]] → [[integration_tests]] → [[system_validation]]
  ```

### Meta-Programming Capabilities

#### Code Generation
```python
def generate_model_code(spec_file: Path) -> str:
    """Generate implementation from specifications.
    See [[ai_documentation_style]] for code generation patterns.
    """
    # Parse markdown specifications
    spec = parse_markdown_spec(spec_file)
    
    # Extract probabilistic model
    model = extract_probabilistic_model(spec)
    
    # Generate implementation
    return generate_implementation(model)
```

#### Validation Rules
```python
def check_probabilistic_consistency():
    """Verify probabilistic consistency.
    See [[validation_framework]] for validation rules.
    """
    # Check matrix constraints
    verify_stochastic_matrices()
    
    # Validate probability measures
    verify_measure_consistency()
    
    # Check inference specifications
    verify_inference_methods()
```

### Benefits

1. **Theoretical Consistency**
   - [[ai_validation_framework|Automated validation]] of mathematical relationships
   - Enforcement of probabilistic constraints
   - Verification of implementation patterns

2. **Learning Support**
   - [[research_education|Guided exploration]] of concepts
   - Clear dependency tracking
   - Interactive knowledge discovery

3. **Implementation Quality**
   - [[ai_documentation_style|Automated code generation]]
   - Consistent design patterns
   - [[testing_guide|Rigorous testing framework]]

4. **Documentation Integration**
   - [[documentation_standards|Living documentation]]
   - [[package_documentation|Executable specifications]]
   - [[ai_validation_framework|Automated validation]]

## Getting Started

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Configure Project**
   - Edit `config.yaml` for project settings
   - Customize templates in `templates/`
   - Set up [[obsidian_linking|Obsidian vault integration]]

3. **Create Cognitive Models**
   - Use [[ai_concept_template|templates]] to define agents
   - Configure belief networks
   - Set up observation spaces
   - Define action policies

4. **Run Simulations**
   - Execute model simulations
   - Analyze results
   - Visualize networks

## Testing

See [[testing_guide]] for comprehensive testing documentation.

### Running Tests
```bash
# Run all tests with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_matrix_ops.py -v

# Run tests with coverage report
python -m pytest --cov=src
```

### Test Organization
- `tests/test_matrix_ops.py`: Matrix operation tests
- `tests/test_visualization.py`: Visualization component tests
- `tests/conftest.py`: Shared test fixtures and configuration

## Development

### Contributing
See [[contribution_guide]] for detailed contribution guidelines.

### Documentation
- [[api_documentation|API Documentation]]
- [[documentation_guide|User Guide]]
- [[example_writing|Examples]]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Active Inference research community
- Obsidian development team
- Contributors and maintainers




