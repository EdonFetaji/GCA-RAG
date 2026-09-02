# ADR 0004: Extraction is open-vocabulary; the ontology moves to the grounder

## Status
Accepted. **Amends ADR 0002** — supersedes its "the ontology every agent in the
pipeline is constrained by" decision. The enums survive; what changes is who
reads them, and when.

## Context
ADR 0002 gave every agent the same `OntologyConfig`: eight entity types, ten
relation types, rendered into the extractor's and grader's prompts as an
allowed list, and enforced a second time by `EntityType` / `RelationType` on the
Pydantic models. A type outside the list could not be extracted, and could not
have been represented if it had been.

A run over Multi-News cluster 0 showed what that costs. The documents describe
eleven governors' races. The extracted graph made every person-to-state edge
`LOCATED_IN`:

- `ent_rick_scott → ent_florida` — he governs Florida
- `ent_pat_mccrory → ent_north_carolina` — he is a candidate there
- `ent_rob_mckenna → ent_washington` — he is the Attorney General
- `ent_ovide_lamontagne → ent_new_hampshire` — he is running in a race there

Four different relationships, one label, because `GOVERNOR_OF` was not on the
list and `LOCATED_IN` was the nearest permitted thing. The information was in
the documents, was read correctly by the model, and was discarded at the schema
boundary.

The loop could not catch it. Round 1's grader came close — *"Suggests ontology
misuse; LOCATED_IN is for persons located in places, RELATED_TO is vague"* — and
the repair pass resolved the complaint by making every such edge `LOCATED_IN`
uniformly. Round 3 reported no defects. Convergence was reached on a graph that
was *consistently* wrong, because the grader's `WRONG_TYPE` check tested
membership of the allowed list, which uniformity satisfies perfectly.

This is not a prompt bug. A closed vocabulary chosen before the corpus is read
cannot express what an arbitrary corpus says, and a grader that checks
conformance to that vocabulary will score the resulting information loss as
success.

## Decision
**The extractor names its own types.** No entity or relation type list reaches
it. `ExtractionTask` no longer carries an `OntologyConfig`; it carries only
`domain_context`, a subject-matter steer that constrains nothing about types.
The v2 system prompt asks for the most specific label the evidence supports and
tells the model to reuse its own labels once chosen.

**`Entity.type` and `Relation.relation_type` become `str`.** The enum on the
model was the second enforcement point and had to go with the first. What
remains enforced is spelling: `normalize_type` folds `located in`,
`Located-In` and `LOCATED_IN` to one token, so `Relation.key` stays stable and
de-duplication downstream still works. Free-form is not the same as unchecked —
an unknown type is a legitimate finding, an inconsistently spelled one never is.

**The grader audits the vocabulary instead of policing it.** It receives the
type vocabulary the graph actually uses, and `WRONG_TYPE` is re-specified around
four failures that a list check cannot see: *imprecise* (a label vaguer than the
evidence supports — the `LOCATED_IN` case above), *overloaded* (one label over
several distinct relationships), *redundant* (two labels for one concept), and
*inapt*. The v2 prompt also warns that uniformity is not consistency, which is
the specific way round 3 was fooled.

**The grounder is the only ontology consumer.** It still receives
`cfg.ontology`, but as a *target* to map onto rather than a constraint applied
upstream. Its hint tables are now keyed by the types present in the graph, so
`GOVERNOR_OF` gets an empty hint list — which the prompt already defines as
"search for it yourself" — while `LOCATION` still gets `dbo:Place` and friends.
If grounding is disabled, no ontology is applied to the run at all, which is the
honest description of what a run without grounding produces.

**Both regimes stay runnable.** v1 templates remain on disk. `KG_PROMPT_VERSION`
defaults to `v2`; setting it to `v1` reproduces the constrained pipeline against
the same code, which is what makes the two comparable on the same clusters.

## Consequences
**Type drift across clusters is now possible and is not automatically a bug.**
Cluster 0 may produce `GOVERNOR_OF` and cluster 7 `IS_GOVERNOR_OF`. Within one
graph the grader polices this; across graphs nothing does. Anything that
aggregates clusters — the Track 3 validator especially — must either canonicalise
through the grounder's DBpedia mappings or treat the local type as a free text
feature. Grounding was optional before; for cross-cluster work it is now the
mechanism that makes graphs commensurable.

**`kg_dataset/corruption.py` samples replacement types from the enums.** It still
runs, but on an open-vocabulary graph it will substitute a type from a
vocabulary the graph does not use, which makes the corruption trivially
detectable and the negative sample too easy. It should sample from the graph's
own vocabulary instead. Not changed here; flagged as the next thing to fix.

**HDF5 stores a wider set of type strings.** The dtype was already a variable
string, so the format is unchanged, but a reader that assumed the enum's eight
values is now wrong.

**Comparison is the point.** The claim that open-vocabulary extraction produces
better graphs is not established by this ADR — it is made testable by it. Run
both prompt versions over the same clusters and compare.

## Revisit if
- Type drift across clusters proves worse than the information loss it replaced,
  and grounding does not reconcile it in practice.
- The grader's re-specified `WRONG_TYPE` turns out to fire constantly, which
  would mean the extractor needs a vocabulary floor rather than none at all — a
  seed list it may extend, rather than a closed list it may not.
