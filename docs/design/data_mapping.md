# Data Mapping

## Goal

Map external ticket datasets into the project's phase-1 `ticket_schema_v1` format.

## Mapping Strategy

Use conservative, rule-based mapping first.

Principles:

- prefer stable mappings over aggressive inference
- keep uncertain fields as `null`
- preserve source lineage in metadata
- make heuristics easy to inspect and revise

## Source 1: `Console-AI/IT-helpdesk-synthetic-tickets`

Raw fields:

- `id`
- `subject`
- `description`
- `priority`
- `category`
- `createdAt`
- `requesterEmail`

### Mapping

- `ticket_id` <- `id`
- `summary` <- normalized `subject`
- `category` <- mapped from raw `category`
- `priority` <- normalized raw `priority`
- `requires_followup` <- `true`
- `reporter.name` <- local part of `requesterEmail`
- `reporter.team` <- `null`
- `affected_systems` <- one object using mapped category and subject cues
- `actions_requested` <- one action derived from subject/description
- `constraints.environment` <- inferred from keywords when possible, else `null`
- `constraints.blocking` <- inferred from urgency / outage terms, else `null`

## Source 2: `KameronB/synthetic-it-callcenter-tickets`

Raw fields include:

- `number`
- `type`
- `contact_type`
- `short_description`
- `content`
- `category`
- `subcategory`
- `customer`
- `software/system`
- `assignment_group`
- `issue/request`

### Mapping

- `ticket_id` <- `number`
- `summary` <- normalized `short_description`
- `category` <- mapped from raw `type` + `subcategory` + `category`
- `priority` <- inferred from `type`, `subcategory`, and text cues
- `requires_followup` <- `true`
- `reporter.name` <- normalized `customer`
- `reporter.team` <- `assignment_group`
- `affected_systems` <- `software/system` plus raw `category` / `subcategory`
- `actions_requested` <- one action derived from `issue/request` or `content`
- `constraints.environment` <- inferred from text when possible
- `constraints.blocking` <- inferred from incident/error/outage cues

## Category Mapping

Project categories:

- `bug`
- `feature`
- `question`
- `incident`
- `task`

### High-level Rules

- requests / upgrades / installations -> usually `feature` or `task`
- incidents / crashes / failures / errors -> `bug` or `incident`
- access / account help requests -> `task`
- explicit questions -> `question`

## Complexity Bucket Heuristic

Assign complexity from mapped content:

### `simple`

- no explicit team
- no environment or blocking signal
- one affected system at most
- one action at most
- shorter summary

### `medium`

- nested values populated
- one action plus richer context

### `complex`

- multiple affected systems or actions
- blocking / environment inferred
- richer structured fields populated

## Known Limitations

- actions are heuristic, not gold extraction
- environment inference is keyword-based
- some source datasets do not contain enough information for strong nested fields
- early mapped data is for pipeline building, not final benchmark quality

## Intended Use

Use mapped external data as:

- initial weakly supervised training data
- schema-shaping material
- seed for later manual cleaning and synthetic expansion
