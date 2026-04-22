# UI Tools Configuration Guide

This directory contains the UI tools system for dynamically selecting and dispatching specialized UI components based on AI agent responses. This is a generic mechanism that can be integrated into any user interface.

## Table of Contents

1. [Overview](#overview)
2. [Defining UI Tools](#defining-ui-tools)
3. [Publishing UI Tools](#publishing-ui-tools)
4. [Specifying Tools Filter via WebSocket](#specifying-tools-filter-via-websocket)
5. [Preprocessed Tools](#preprocessed-tools)
6. [Tool Validation](#tool-validation)
7. [Schema Properties & Enums](#schema-properties--enums)
8. [Prompt Keywords](#prompt-keywords)
9. [Field Validation Rules](#field-validation-rules)
10. [Example: Complete Tool Definition](#example-complete-tool-definition)
11. [Best Practices](#best-practices)

---

## Overview

UI tools allow the AI agent to select and dispatch specialized UI components based on the context of user requests. Each tool is associated with:

- **Name**: Unique identifier for the tool
- **Description**: What the tool does
- **Prompt**: Instructions for the LLM on when and how to use this tool
- **Category**: Type of component (editor, route, console, selector, etc.)
- **Schema**: JSON schema defining required and optional parameters
- **Metadata**: Version, compatibility information, and enum definitions

---

## Defining UI Tools

### Basic Tool Structure

```json
{
  "name": "tool-identifier",
  "description": "Brief description of what this tool does",
  "prompt": "Instructions for the LLM about when to use this tool",
  "category": "selector|editor|any-other-category",
  "revision": 1,
  "enabled": true,
  "defaultValues": {
    "enabled": true
  },
  "schema": {
    "properties": {
      "fieldName": {
        "type": "string|number|boolean|object",
        "description": "Field description",
        "required": true
      }
    }
  },
  "metadata": {
    "version": "1.0",
    "any-other-metadata": "value"
  }
}
```

### Tool Naming Conventions

- Use lowercase, hyphen-separated names: `display-json`, `open-editor`, `show-diff`
- Names should clearly indicate purpose
- Avoid ambiguous names; namespace them (e.g., `resource-viewer`, `config-editor`)

### Writing Effective Tool Prompts

The prompt is critical for LLM decision-making. It should:

1. **Define scope clearly** (use `TOOL SCOPE:` marker):
   ```
   TOOL SCOPE: single-resource
   TOOL SCOPE: list or collection
   TOOL SCOPE: navigation
   ```

2. **Specify use cases** (when to use):
   ```
   USE THIS TOOL ONLY WHEN:
   - The context names a specific resource by exact name
   - The user is asking for details about that ONE resource
   ```

3. **Prohibit misuse** (when NOT to use):
   ```
   NEVER USE THIS TOOL WHEN:
   - Showing multiple items (list of configs, users, etc.)
   - The context mentions 'list', 'all', 'multiple', or plurals
   - You would need to call this tool 2+ times with different item identifiers
   ```

4. **Include call instructions** (how to invoke):
   ```
   When you call this tool: Display formatted content for item {resourceId} 
   of type {resourceType} in an appropriate viewer. The content is: {content}. 
   PRESERVE all formatting, structure, and encoding.
   ```

---

## Publishing UI Tools

UI tools are published via Kubernetes **ConfigMaps** . The system automatically discovers and loads ConfigMaps on each request.

### ConfigMap Format

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ui-tools-config
  namespace: default
  labels:
    app: rancher-ai-ui-tools  # Required label for discovery
  annotations:
    description: "Defines UI tools"
data:
  config.json: |
    {
      "metadata": {
        "annotations": {
          "description": "Defines UI tools"
        }
      },
      "config": {
        "systemPrompt": "You are a helpful assistant for selecting UI components...",
        "enabled": true,
        "maxTools": 5,
        "revision": 1
      },
      "tools": [ /* tool definitions */ ]
    }
```

### Publication Process

1. **Define tools** in ConfigMap `data.config.json`
2. **Apply ConfigMap** to cluster:
   ```bash
   kubectl apply -f ui-tools-configmap.yaml
   ```
3. **ConfigMap loads** ConfigMaps automatically loaded on user's request
4. **Tools available** to AI agent for selection in subsequent requests

### Configuration Section

The `config` object controls global settings for the entire tool set:

```json
{
  "config": {
    "systemPrompt": "You are a helpful assistant for selecting UI components...",
    "enabled": true,
    "maxTools": 5,
    "revision": 1
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `systemPrompt` | string | Yes | System instruction provided to the LLM for how to think about UI tool selection. Guides the agent on selection strategy and best practices. |
| `enabled` | boolean | No | Whether this entire tool configuration is active. Set to `false` to temporarily disable all tools in this ConfigMap without deleting it. Default: `true` |
| `maxTools` | number | No | Maximum number of tools the LLM can select for a single response. Prevents tool overload and keeps responses concise. Default: `10` |
| `revision` | number | No | Version counter for tracking configuration changes. Increment when updating tools or config to invalidate caches. Default: `1` |

### Configuration Scoping

Tools are scoped by ConfigMap name (e.g., `default`, `prod-cluster`):
- Different ConfigMaps can define different tool sets
- Request metadata specifies which config to use
- Prevents tool name collisions across teams

---

## Specifying Tools Filter via WebSocket

UIs can filter which tools are available to the AI agent by sending a tools filter in the WebSocket request. This allows fine-grained control over which UI components can be dispatched for specific requests.

### WebSocket Request Format

Send a JSON request with a `tools` object containing:
- **name**: The ConfigMap name where tools are defined (e.g., `default`, `prod-cluster`)
- **tools**: Array of tool names to include (only these tools will be selected by the agent)

```json
{
  "prompt": "Show me the resource details",
  "tools": {
    "name": "default",
    "tools": ["display-config", "compare-versions", "open-editor"]
  },
  "agent": "my-agent"
}
```

### How Tool Filtering Works

1. **Available Tools**: The agent loads all tools from the specified ConfigMap (`name` field)
2. **Filter Applied**: Only tools in the `tools` array are considered for selection
3. **Disabled Tools Excluded**: Even if listed, disabled tools are skipped
4. **Selection**: The AI agent's UI tools selector chooses from the filtered set

### Example Scenarios

#### Scenario 1: Restrict to Viewing Tools

UI is displaying a resource details view and wants only viewing/comparison tools:

```json
{
  "prompt": "Show me the current configuration",
  "tools": {
    "name": "default",
    "tools": ["display-config", "compare-versions", "export-data"]
  }
}
```

**Result**: Agent can only dispatch viewing and comparison tools, not navigation or modification tools.

#### Scenario 2: Restrict to Navigation Tools

UI is in list/collection view and wants only navigation/discovery tools:

```json
{
  "prompt": "Which items match the criteria?",
  "tools": {
    "name": "default",
    "tools": ["navigate-to-item", "open-search", "filter-results"]
  }
}
```

**Result**: Agent can only dispatch navigation and discovery tools for exploring collections.

#### Scenario 3: Empty Tools List (Disable All)

To prevent any UI tools from being dispatched:

```json
{
  "prompt": "What is the status of my cluster?",
  "tools": {
    "name": "default",
    "tools": []
  }
}
```

**Result**: No UI tools will be selected, only text response is sent.

#### Scenario 4: Context-Specific Tools

Different UI sections enable different tool combinations:

```json
{
  "prompt": "Show details for resource-app-prod-01",
  "tools": {
    "name": "default",
    "tools": [
      "display-config",
      "compare-versions",
      "open-logs",
      "select-action"
    ]
  }
}
```

**Result**: Agent can choose from viewing, comparison, debugging, or interaction tools depending on response context.

### Error Handling

- **Missing ConfigMap**: If the specified `name` doesn't exist, UI tools dispatch is skipped
- **Empty Filter**: If `tools` array is empty, no UI tools are dispatched
- **Invalid Tool Names**: Non-existent tool names in the filter are silently ignored
- **Disabled Tools**: Disabled tools won't be selected even if listed in the filter

### Best Practices

1. **Start Broad, Refine**: Begin with all relevant tools, then restrict as needed
2. **Match Context**: Use filters that match the current UI view (single resource vs. list)
3. **Disable Intentionally**: Use empty `tools: []` array only when UI tools would be confusing
4. **ConfigMap Organization**: Use different ConfigMaps (`prod`, `dev`, `test`) with different tool sets per environment
5. **Document Filters**: Specify in UI code which tools are expected for each view type

---

## Preprocessed Tools

### Built-in Preprocessed Tools

The system provides two preprocessed tools automatically on Rancher confirmation actions (create/update/patch):

#### 1. **show-yaml** (Editor Category)

**Purpose**: Display YAML content of a single Kubernetes resource to be created.

**When Produced**: 
- Automatically available when resource content is available
- Used when displaying configuration of one specific resource

**Parameters**:
```json
{
  "yaml": "complete, unmodified YAML content",
  "resourceKind": "Pod|Deployment|Service|etc.",
  "resourceName": "exact resource name",
  "resourceNamespace": "namespace of resource",
  "copyable": true,
  "editable": true
}
```

#### 2. **show-yaml-diff** (Editor Category)

**Purpose**: Display side-by-side YAML diff between current and proposed states of a resource to be updated/patched.

**When Produced**:
- Available when showing before/after changes to ONE resource
- Used during resource modification, patching, or updates

**Parameters**:
```json
{
  "original": "current/existing YAML unchanged",
  "patched": "proposed/new YAML unchanged",
  "resourceKind": "Pod|Deployment|etc.",
  "resourceName": "exact resource name",
  "resourceNamespace": "namespace",
  "copyable": true,
  "editable": true
}
```

---

## Tool Validation

Validation occurs at multiple levels:

### 1. **Category-Based Validation**

Different categories enforce different rules - only selector category has validation at this time:

| Category | Purpose | Max Options | Requires Selection |
|----------|---------|-------------|-------------------|
| `selector` | User picks option | 3 | Yes |

### 2. **Required Fields Validation**

Schema validation ensures all required fields are present:

```json
{
  "properties": {
    "yaml": {
      "type": "string",
      "description": "The YAML content",
      "required": true  // ← Must be provided by LLM
    },
    "copyable": {
      "type": "boolean",
      "description": "Enable copy button"
      // optional, no "required" field
    }
  }
}
```

**Validation Logic**:
- Fields marked `"required": true` must be present
- Field types must match schema (`string`, `boolean`, `number`, etc.)
- Missing required fields → tool dispatch rejected
- Type mismatch → tool dispatch rejected

### 3. **Enum Validation**

If tool property defines enum values, LLM must select from allowed list:

```json
{
  "viewType": {
    "type": "string",
    "enum": ["list", "detail", "timeline", "chart"],
    "required": true
  }
}
```

**Validation**:
- LLM must provide one of: `list`, `detail`, `timeline`, or `chart`
- Any other value → validation fails
- See [Enums](#schema-properties--enums) section below

### 4. **String Constraints Validation**

Additional validation for string properties:

```json
{
  "label": {
    "type": "string",
    "maxLength": 50,
    "description": "Display text"
  }
}
```

**Constraints**:
- `maxLength`: Maximum allowed string length (e.g., 50 chars)
- `minLength`: Minimum required length
- Both are checked during validation

---

## Schema Properties & Enums

### Type Definitions

```json
{
  "yaml": {
    "type": "string",           // Text content
    "description": "YAML content"
  },
  "copyable": {
    "type": "boolean",          // true/false
    "description": "Enable copy"
  },
  "timeout": {
    "type": "number",           // Integer or decimal
    "description": "Wait time (seconds)"
  },
  "metadata": {
    "type": "object",           // Nested JSON object
    "description": "Custom metadata"
  }
}
```

### Enum Properties

Enums restrict values to a predefined list:

```json
{
  "viewType": {
    "type": "string",
    "enum": ["list", "detail", "timeline", "chart"],
    "description": "Display format"
  }
}
```

### Enum with Metadata (Advanced)

Use metadata for detailed enum information. **Important**: Fields in the `requires` array must exist as properties in the schema:

```json
{
  "schema": {
    "properties": {
      "viewType": {
        "type": "string",
        "enum": ["list", "detail", "timeline", "chart"],
        "description": "Choose view type"
      },
      "filter": {
        "type": "string",
        "description": "Filter criteria"
      },
      "itemId": {
        "type": "string",
        "description": "Item identifier"
      },
      "dateRange": {
        "type": "object",
        "description": "Date range for timeline"
      },
      "metricType": {
        "type": "string",
        "description": "Type of metric to display"
      }
    }
  },
  "metadata": {
    "enum": {
      "viewType": {
        "list": {
          "description": "SCOPE: collection. Shows all items in a tabular list.",
          "requires": ["filter"]
        },
        "detail": {
          "description": "SCOPE: single. Shows detailed view of one item.",
          "requires": ["itemId"]
        },
        "timeline": {
          "description": "SCOPE: collection. Shows time-series events.",
          "requires": ["dateRange"]
        },
        "chart": {
          "description": "SCOPE: collection. Shows statistics/metrics visualization.",
          "requires": ["metricType"]
        }
      }
    }
  }
}
```

**Enum Features**:
- Each enum value can have a description
- Each can specify required parameters
- Helps LLM understand when to use each option
- Displayed in UI for user selection

---

## Prompt Keywords

The prompt field uses special keywords and markers to guide LLM behavior:

### Scope Keywords

Mark the scope explicitly:

```
TOOL SCOPE: single-resource       ← For one specific resource
TOOL SCOPE: list or collection    ← For multiple resources or lists
TOOL SCOPE: navigation            ← For page navigation
TOOL SCOPE: configuration         ← For settings/config
```

### Action Keywords

Guide when the LLM should use the tool:

```
USE THIS TOOL ONLY WHEN:
- Exact condition 1
- Exact condition 2

NEVER USE THIS TOOL WHEN:
- Prohibited condition 1
- Prohibited condition 2
```

### Tool Selection Context Keywords

These keywords are used in the internal UI tools selector prompt structure to provide context for LLM-based tool selection:

```
SELECTED AGENT    ← The agent configuration used to perform the user's request
                     Includes: agent name, description
                     Used for: scoping tool selection to agent's capabilities

CONTEXT          ← The current response/context from the agent node
                    Includes: assistant message, state information
                    Used for: understanding what tools would enhance this specific response

MCP RESPONSE     ← Raw response from MCP calls (if available)
                    Includes: structured data from external tools
                    Used for: better tool parameter extraction and accuracy
                    Optional: only included when MCP data is available
```

The selector uses these sections to determine which tools would best enhance the response by understanding:
- **Agent capability** (what this agent specializes in)
- **Current context** (what the agent is responding about)
- **Available data** (raw MCP data for accurate parameter extraction)

Use these keywords in the tool prompt to instruct the LLM on when and how to select the tool based on the response context.

### Example Prompt with Keywords

```
TOOL SCOPE: single-resource

You are selecting whether to use this tool. THIS TOOL IS FOR ONE SPECIFIC ITEM ONLY.

USE THIS TOOL ONLY WHEN:
- The context identifies a specific item by exact ID or name (e.g., 'config-prod-db', 'user-12345')
- The user is asking for details about that ONE item
- You are displaying information about a single resource

NEVER USE THIS TOOL WHEN:
- Showing multiple items (list of configs, users, etc.)
- The context mentions 'list', 'all', 'multiple', 'summary', or plurals
- You would need to call this tool 2+ times with different item identifiers

If you're tempted to call this tool multiple times with different item IDs, STOP. 
That means there's a collection/list tool you should use instead.

When you call this tool: Display formatted content for item {resourceId} 
of type {resourceType} in an appropriate viewer. The content is: {content}. 
PRESERVE all formatting, structure, and encoding.
```

---

## Field Validation Rules

### By Type

| Type | Validation |
|------|-----------|
| `string` | Length constraints (maxLength, minLength), enum validation |
| `boolean` | Must be true/false |
| `number` | Integer or decimal values |
| `object` | JSON structure validation |
| `array` | Item type validation (if specified) |

### By Requirement

```json
{
  "required": true    // Must be provided in tool call
  // (absent) = optional field
}
```

### By Constraint

```json
{
  "enum": ["a", "b"], // Allowed values only
}
```

---

## Example: Complete Tool Definition

```json
{
  "name": "display-json",
  "description": "Display formatted JSON content of a resource",
  "prompt": "TOOL SCOPE: single-resource\n\nYou are selecting whether to use this tool...",
  "category": "viewer",
  "revision": 1,
  "enabled": true,
  "defaultValues": {
    "enabled": true
  },
  "schema": {
    "properties": {
      "content": {
        "type": "string",
        "description": "JSON content to display",
        "required": true
      },
      "resourceType": {
        "type": "string",
        "description": "Type of resource (Config, Database, etc.)",
        "required": true
      },
      "resourceId": {
        "type": "string",
        "description": "Exact resource identifier",
        "required": true,
        "maxLength": 500
      },
      "title": {
        "type": "string",
        "description": "Display title",
        "required": false
      },
      "searchable": {
        "type": "boolean",
        "description": "Enable search functionality"
      },
      "copyable": {
        "type": "boolean",
        "description": "Enable copy to clipboard"
      },
      "viewType": {
        "type": "string",
        "enum": ["list", "detail", "timeline"],
        "description": "Display format for the content",
        "required": true
      }
    }
  },
  "metadata": {
    "version": "1.0",
    "enum": {
      "viewType": {
        "list": {
          "description": "SCOPE: collection. Shows items in tabular list.",
          "requires": ["searchable"]
        },
        "detail": {
          "description": "SCOPE: single. Shows detailed view of one item.",
          "requires": ["resourceId"]
        },
        "timeline": {
          "description": "SCOPE: collection. Shows time-series events.",
          "requires": ["resourceId"]
        }
      }
    }
  }
}
```

---

## Best Practices

1. **Prompts**: Be explicit about scope (single vs. multiple resources)
2. **Required Fields**: Only mark fields as required if always needed
3. **Descriptions**: Provide clear, actionable descriptions for all fields
4. **Constraints**: Use maxLength for UI display limits
5. **Enums**: List all valid options for route/selector tools
6. **Naming**: Use clear, namespaced names for tools
7. **Validation**: Ensure schema matches actual implementation

