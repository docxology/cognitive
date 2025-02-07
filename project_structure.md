# Project Structure

## Core Directories

### 📁 templates/
- `node_templates/` - Base templates for different types of cognitive nodes (See [[ai_concept_template]])
  - `agent_template.md` - Template for [[active_inference]] agent definitions
  - `belief_template.md` - Template for [[belief_updating]] structures
  - `goal_template.md` - Template for goal hierarchies and optimization
  - `action_template.md` - Template for [[action_selection]] definitions
  - `observation_template.md` - Template for [[predictive_processing]] patterns
  - `relationship_template.md` - Template for [[linking_completeness|node relationships]]

### 📁 knowledge_base/
- `cognitive/` - Core cognitive concepts (See [[cognitive_phenomena]])
  - [[active_inference]] - Active Inference framework
  - [[free_energy_principle]] - Free Energy Principle
  - [[predictive_processing]] - Predictive Processing theory
- `agents/` - Agent definitions and states
- `beliefs/` - Belief networks and structures
- `goals/` - Goal hierarchies and definitions
- `actions/` - Action repertoires and policies
- `observations/` - Observation patterns and histories
- `relationships/` - Inter-node relationships and dynamics

### 📁 src/
- `models/` - Core modeling components
  - `active_inference/` - [[active_inference|Active Inference]] implementation
  - `belief_updating/` - [[belief_updating|Belief update mechanisms]]
  - `policy_selection/` - [[action_selection|Policy selection algorithms]]
  - `state_estimation/` - State estimation tools
- `utils/` - Utility functions and helpers
  - `visualization/` - [[visualization_tools|Visualization tools]]
  - `data_processing/` - Data processing utilities
  - `obsidian_integration/` - [[obsidian_linking|Obsidian integration tools]]
- `analysis/` - Analysis tools and scripts
  - `network_analysis/` - Network analysis tools
  - `metrics/` - Performance and behavior metrics
  - `simulations/` - Simulation frameworks

### 📁 docs/
See [[documentation_standards]] for comprehensive documentation guidelines

#### Core Documentation
- `api/` - API Reference Documentation
  - [[api_documentation]] - API documentation guidelines
  - [[api_reference]] - Complete API reference
  - [[api_versioning]] - API versioning guidelines
  - [[api_examples]] - API usage examples

#### Conceptual Documentation
- `concepts/` - Core Concepts and Theory
  - [[machine_readability]] - Machine processing principles
  - [[plain_text_benefits]] - Benefits of plain text
  - [[research_education]] - Research and education integration
  - [[cognitive_phenomena]] - Cognitive phenomena catalog
  - [[theoretical_foundations]] - Theoretical background

#### Implementation Guides
- `guides/` - Usage Guides and Tutorials
  - [[ai_documentation_style]] - Documentation style guide
  - [[ai_file_organization]] - File organization guide
  - [[ai_folder_structure]] - Directory structure guide
  - [[ai_semantic_processing]] - Semantic processing guide
  - [[ai_validation_framework]] - Validation framework
  - [[linking_completeness]] - Link completeness guide
  - [[linking_analysis]] - Link analysis guide
  - [[linking_validation]] - Link validation guide
  - [[obsidian_linking]] - Obsidian integration guide
  - [[package_documentation]] - Package documentation guide

#### Example Documentation
- `examples/` - Example Implementations
  - [[example_writing]] - Example writing guide
  - [[quickstart_example]] - Quick start tutorial
  - [[active_inference_example]] - Active inference examples
  - [[belief_updating_example]] - Belief updating examples
  - [[integration_examples]] - Integration examples
  - [[validation_examples]] - Validation examples

#### Template Documentation
- `templates/` - Documentation Templates
  - [[package_component]] - Package component template
  - [[ai_concept_template]] - AI concept template
  - [[linking_template]] - Linking template
  - [[validation_template]] - Validation template
  - [[example_template]] - Example template
  - [[guide_template]] - Guide template

#### Tool Documentation
- `tools/` - Development Tools
  - [[documentation_tools]] - Documentation tooling
  - [[validation_tools]] - Validation tools
  - [[visualization_tools]] - Visualization tools
  - [[analysis_tools]] - Analysis tools
  - [[generation_tools]] - Code generation tools

#### Research Documentation
- `research/` - Research Documentation
  - [[experiment_documentation]] - Experiment documentation
  - [[result_analysis]] - Result analysis
  - [[literature_review]] - Literature review
  - [[theoretical_development]] - Theory development

### 📁 tests/
- Unit tests and integration tests (See [[testing_guide]])
  - `test_matrix_ops.py` - Matrix operation tests
  - `test_visualization.py` - Visualization tests
  - `conftest.py` - Shared test fixtures

### 📁 data/
- `raw/` - Raw data storage
- `processed/` - Processed data
- `results/` - Analysis results

## Key Files

- `requirements.txt` - Python dependencies
- `setup.py` - Package setup
- `README.md` - Project documentation (See [[documentation_standards]])
- `CONTRIBUTING.md` - [[contribution_guide|Contribution guidelines]]
- `.gitignore` - [[git_workflow|Git ignore rules]]
- `config.yaml` - Configuration settings

## Documentation Integration

### Knowledge Graph Structure
The project follows a [[linking_completeness|comprehensive linking structure]] that ensures:
- Bidirectional relationships between components
- Clear dependency chains
- Traceable implementations
- Validated connections

### Validation Framework
See [[ai_validation_framework]] for details on:
- Link validation
- Documentation completeness
- Quality metrics
- Automated checks

### Best Practices
Follow these guides for development:
- [[ai_documentation_style]] - Documentation style guide
- [[ai_file_organization]] - File organization patterns
- [[ai_folder_structure]] - Directory structure guidelines
- [[ai_semantic_processing]] - Semantic processing integration 