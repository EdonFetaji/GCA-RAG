# ADR 0003: The grounder is a tool-calling agent

## Status
Accepted. **Amends ADR 0002** — supersedes its "the grounder disambiguates; it
does not retrieve" decision. Everything else in 0002 stands.

## Context
ADR 0002 built the grounder as two phases: `_gather_candidates` fetched a fixed
number of candidates per entity through the `GroundingBackend` port, and one LLM
call chose among them. It named the conditions to revisit that, and both were
met before the design ever ran:

1. **The deciding fact is one hop past the candidate list.** A candidate row
   carries a URI, a label, and a popularity count. What separates the city from
   the band of the same name is the resource's abstract and types — which is a
   second lookup the pre-fetch design has no way to make, because it is chosen
   based on the first one's answer.
2. **Relations were not really grounded at all.** `resolve_property` mapped a
   local relation type to a property URI by table lookup, entity-independently.
   Whether DBpedia actually asserts an edge between the two grounded endpoints —
   the strongest evidence available — was never asked.

There was also nothing to revisit against: the server ADR 0002 specified,
`mcp_servers/dbpedia/`, was never written, and `mcp_backend.py` imported `Client`
from `mcp` (where it does not exist) rather than `fastmcp`. The whole grounding
path was dead, failing silently into "no candidates".

## Decision
The grounder becomes a tool-calling agent over a real DBpedia MCP server.

**Eight tools, in two groups.** Entity: `spotlight_link` (context-aware linking),
`search_resource` (name search), `get_resource_profile` (confirmation),
`search_class` (class discovery). Relation: `find_object_properties`,
`find_datatype_properties`, `get_property_profile`, `get_predicates_between`.
They live in `mcp_server/`, which is where the scaffold already was — the
`mcp_servers/dbpedia/` name in ADR 0002 was aspirational and is now retired.

**The output contract is unchanged.** `GroundedKnowledgeGraph` still wraps the
graph untouched and carries mappings alongside it. Grounding is still additive:
a failure degrades to "ungrounded", never to a corrupted graph.

## Decisions worth recording

**The server knows nothing about the pipeline's ontology.** `mcp_server/` speaks
DBpedia only. The `PERSON → dbo:Person` table lives in
`kg_agentic_extraction/grounding/hints.py` and is rendered into the prompt as
*unverified starting guesses*, not applied as a rewrite. A wrong guess is then
visible in a prompt and correctable by the model with `search_class`, instead of
being a silent substitution inside a server the pipeline does not own.

**No tool raises.** Every tool returns an empty result plus a `note` explaining
why — unreachable endpoint, rejected URI, no match. An exception propagating out
of a tool tears down the agent's loop mid-graph and discards a transcript that
cost several round-trips; a note the model can read lets it take another route.
`dispatch()` upholds the same contract for calls the model itself got wrong.

**Tool binding is a separate protocol.** `llm/base.py` already said capabilities
beyond `structured()` belong on a protocol only the agents needing them depend
on. `ToolCallingLLMClient` is that protocol, and the extractor and grader keep
depending on the narrower `LLMClient`. The whole loop sits behind one method
rather than exposing message bookkeeping, because that bookkeeping is
provider-specific — the implementation is a single mixin over LangChain, which
all three adapters already wrap.

**The final answer is produced with tools unbound.** Structured output and tool
binding are two competing constraints on one generation. The loop ends with a
separate `with_structured_output` call over the accumulated transcript.

**Sync stays sync.** MCP is async; the nodes, agents, and CLI are not.
`SyncMCPClient` owns a daemon thread running one event loop and holds one
session for the whole grounding pass. That is what makes the pipeline drivable
from a notebook or an async server — the previous `asyncio.run()`-per-call bridge
raised outright there, and paid a handshake per lookup everywhere else.

**The grounder pins its own prompt version.** It resolves `v2` while the
extractor and grader stay on `v1`, via `KG_GROUNDER_PROMPT_VERSION`. A single
shared version would have forced a no-op `v2` of both templates just to move one.
The `v1` grounder templates remain on disk for provenance but are no longer
renderable by the current agent — they expect a `candidate_hints` variable the
pre-fetch design supplied.

## What DBpedia's public endpoints actually do
Found while building this; the tools work around all three.

- **The SPARQL snapshot serves no `dbo:abstract` or `rdfs:comment` for
  resources.** `get_resource_profile` fills the abstract from the Lookup API
  instead, matching on exact URI.
- **`api.dbpedia-spotlight.org` is frequently unreachable.** `spotlight_link`
  returns a note pointing at `search_resource`; `DBPEDIA_SPOTLIGHT_ENDPOINT`
  redirects it at a local container.
- **POST to `dbpedia.org/sparql` does not connect from every network.** All
  queries go over GET.

## Revisit this if
- A grounding pass costs more than the extractor↔grader loop it follows. The
  round budget (`KG_GROUNDING_MAX_TOOL_ROUNDS`) is the first lever; batching
  entities into one prompt each is the second.
- Wikidata replaces or joins DBpedia. That is a second MCP server and a second
  `GroundingBackend`, not a change to the agent.
- The pipeline goes async end-to-end, at which point `SyncMCPClient` is dead
  weight and `MCPClient` is used directly.
