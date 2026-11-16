# Tool catalog

`catalogs/tool_catalog.json` is the public contract for all 116 tools. Each
tool has one name, one functional group, a description, an input schema, an
output schema, and an illustrative input example. All tools use the same
registry and MCP call path.

## Responses

Every successful MCP call returns a JSON object. Tools that produce a JSON
array or scalar expose it under `result`; tools that produce a JSON object
return its fields directly. Provider-backed tools can return different fields
depending on the configured source. Their output schemas describe the stable
envelope and allow provider-specific fields. Tool exceptions are reported as
MCP errors rather than successful payloads.

Some local tools write files. Their response can be a completion object with
`status`, `tool`, and `output_path`; for a path inferred inside the operation,
`output_path` can be null. The output schema and description for each tool
specify whether data is returned directly or saved to a file. External
training and benchmark tools launch a background task and return `task_id`;
poll progress with `training_get_task` and enumerate tasks with
`training_list_tasks`.

## Groups and names

The `groups` section describes functions rather than implementation sources.
Membership comes from each tool's `group` field. The optional
`default_source` is a connector fallback for groups that use group-based
lookup; it does not change the registry or classify every member of a group.

The `replacements` section maps retired public names to current operations.
Clients should call the current names in `tools`. The mapping is documentation
for migration and does not create extra registered tools.

The input examples show valid argument structures. File paths and provider
credentials in examples must be supplied for the user's own environment before
running tools that read local data or call protected APIs.
