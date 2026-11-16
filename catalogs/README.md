# Unified catalog

`tool_catalog.json` is the canonical public index for the 116 MCP tools:

| Section | Content |
| --- | --- |
| `tools` | Public tools. Every record has the same six keys: `name`, `description`, `group`, `input_schema`, `output_schema`, `example`. |
| `groups` | Group descriptions and optional default data sources, keyed by group ID. Tool membership is derived from each tool's `group` field. |
| `replacements` | Retired tool IDs mapped to current public operations. |

A tool belongs to exactly one defined group, and every public tool is included
in the generated groups view. `src/finance_tool_universe/catalog.py` validates
these relationships when loading the catalog.

Every public operation is registered by `financial_tool.registry.ToolRegistry`
through `financial_tool.tools`. The MCP gateway calls one FastMCP tool manager
for all 116 tools. Input schemas in this catalog match the registered callable
schemas. Every tool has a nonempty output schema; source-dependent payloads
allow additional fields while documenting the stable response envelope.

The remaining files have separate purposes: `data_types.json` contains data
schemas, `company_aliases.json` resolves company names, and the two YAML files
define source routing and supported actions. They do not define another tool
list. Response conventions and tool-name mappings are in `docs/CATALOG.md`.
