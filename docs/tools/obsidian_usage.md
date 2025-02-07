# Obsidian Usage Guide

## Overview
Obsidian serves as our knowledge management system, providing powerful linking and visualization capabilities for cognitive modeling networks.

## Vault Structure

### Core Directories
```
📁 knowledge_base/          # Main knowledge repository
├── agents/                # Agent definitions
├── beliefs/              # Belief networks
├── goals/               # Goal hierarchies
├── actions/             # Action repertoires
├── observations/        # Observation patterns
└── relationships/       # Node relationships

📁 templates/              # Template definitions
└── node_templates/      # Node type templates

📁 docs/                  # Documentation
```

## Node Types and Templates

### Core Node Types
- [[agent_template]] - Agent definitions
- [[belief_template]] - Belief structures
- [[goal_template]] - Goal definitions
- [[action_template]] - Action patterns
- [[observation_template]] - Observation models
- [[relationship_template]] - Node relationships

See [[node_types]] for detailed specifications.

## Linking Patterns

### Internal Links
- Use `[[filename]]` for direct links
- Use `[[filename|alias]]` for custom link text
- Reference [[linking_patterns]] for conventions

### Backlinks
- Automatically tracked by Obsidian
- View in right sidebar
- Essential for [[network_analysis]]

## Knowledge Organization

### Tags
Common tags:
- #agent
- #belief
- #goal
- #action
- #observation
- #relationship

See [[tagging_guide]] for conventions.

### YAML Frontmatter
```yaml
---
type: agent
id: "agent_001"
created: 2024-02-05
modified: 2024-02-05
tags: [agent, cognitive-model]
aliases: []
---
```

## Visualization

### Graph View
- Access via Graph View button
- Shows knowledge network structure
- Color-coded by [[node_types]]
- Configurable layouts

### Filters
- Filter by node type
- Filter by tags
- Filter by relationships
- See [[visualization_guide]]

## Integration Features

### With Cursor
- Markdown preview
- Code block syntax highlighting
- Integration with [[cursor_integration]]

### With Git
- Version control integration
- Collaboration features
- See [[git_workflow]]

## Workflows

### Creating New Nodes
1. Use template hotkeys
2. Fill in frontmatter
3. Add content
4. Establish links
See [[workflow_guides]] for details.

### Maintaining Networks
- Regular updates
- Link verification
- Network analysis
- See [[maintenance_guide]]

## Plugins and Extensions

### Core Plugins
- Graph View
- Backlinks
- Tags
- Templates

### Community Plugins
- Dataview (for queries)
- Calendar (for temporal tracking)
- Mind Map (for hierarchies)

## Best Practices

### Organization
- Use consistent naming
- Maintain clean hierarchy
- Follow [[template_guide]]
- Regular [[maintenance_guide]]

### Linking
- Be specific with links
- Use bidirectional linking
- Maintain link context
- Follow [[linking_patterns]]

### Documentation
- Keep notes updated
- Use templates consistently
- Follow [[documentation_guide]]

## Tips and Tricks

### Keyboard Shortcuts
- `Ctrl/Cmd + O` - Quick switcher
- `Ctrl/Cmd + E` - Toggle edit/preview
- `[[` - Create link
- See [[shortcuts_guide]]

### Search
- Use tags for categorization
- Use frontmatter for metadata
- Full-text search available
- See [[search_guide]]

## Troubleshooting

### Common Issues
- Broken links
- Template issues
- Plugin conflicts
See [[troubleshooting]] for solutions.

## Related Guides
- [[getting_started]]
- [[template_guide]]
- [[network_analysis]]
- [[maintenance_guide]]

## References
- [Obsidian Documentation](https://help.obsidian.md)
- [[code_organization]]
- [[contribution_guide]] 