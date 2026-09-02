"""
Tool definitions — the DBpedia grounding toolkit.

Eight tools in two groups. Entity tools resolve a surface form to a
`dbpedia.org/resource/…` URI; relation tools resolve a relation phrase to a
`dbpedia.org/ontology/…` property, and check whether DBpedia actually asserts
the edge.

They are registered onto the server in `server.py` via `register_tools()`.

To add a tool: write the function here, decorate it inside `register_tools`,
done. Nothing in `server.py` needs to change.

Two rules every tool in this file follows:

**Never raise.** A tool exception stalls the agent loop that called it. An
unreachable endpoint, a malformed URI, an empty match — all come back as an
empty result with `note` explaining why, so the model can pick a different tool.

**Never interpolate raw input.** Model-supplied URIs and strings go through
`sparql.uri_term()` / `sparql.text_literal()` before they reach a query.
"""

from __future__ import annotations

import logging
import re

from fastmcp import FastMCP

from mcp_server.dbpedia import lookup, sparql, spotlight
from mcp_server.dbpedia.classes import (
    DBO,
    DBR,
    normalize_classes,
    normalize_datatype,
    short_name,
)
from mcp_server.dbpedia.config import get_settings
from mcp_server.dbpedia.models import (
    ClassHit,
    ClassSearchResult,
    PredicateLink,
    PredicatesBetweenResult,
    PredicateUsage,
    PropertyHit,
    PropertyProfile,
    PropertySearchResult,
    ResourceHit,
    ResourceProfile,
    ResourceSearchResult,
    SpotlightCandidate,
    SpotlightResult,
)

logger = logging.getLogger(__name__)

#: Predicates that say nothing about what a resource *is*, and would crowd out
#: the ones that do in a profile.
_NOISE_PREDICATES = {
    f"{DBO}abstract",
    f"{DBO}wikiPageWikiLink",
    f"{DBO}wikiPageRedirects",
    f"{DBO}wikiPageDisambiguates",
    f"{DBO}wikiPageExternalLink",
    f"{DBO}wikiPageID",
    f"{DBO}wikiPageRevisionID",
    f"{DBO}wikiPageLength",
    f"{DBO}thumbnail",
}

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


# ── Registration ──────────────────────────────────────────────────────


def register_tools(mcp: FastMCP) -> FastMCP:
    """
    Attach every tool in this module to `mcp`.

    Takes the server as an argument instead of importing it, so that
    `server.py` stays the single place that owns the server instance and there
    is no circular import between the two files.
    """

    # ══ Entity tools ══════════════════════════════════════════════════

    @mcp.tool()
    def spotlight_link(mention: str, context: str) -> SpotlightResult:
        """
        Link a mention to DBpedia resources using the sentence it appears in.

        Reach for this FIRST whenever you have the surrounding text. It is the
        only tool that reads context, so it is what separates "Jordan" the
        country from "Jordan" the basketball player, or a technology company
        from a medieval bishop of the same name.

        Pass the whole sentence (or a few sentences) as `context`, not just the
        mention. Candidates come back ranked, with a contextual score saying how
        well the context supports each one.

        If this returns nothing — the linking service is unreliable and its
        `note` will say so — fall back to `search_resource`.
        """
        logger.info("spotlight_link(%r)", mention)
        settings = get_settings()

        text = (context or "").strip() or (mention or "").strip()
        if not text:
            return SpotlightResult(mention=mention, note="no mention or context given")

        try:
            rows = spotlight.candidates(text, settings=settings)
        except spotlight.SpotlightError as exc:
            logger.warning("spotlight unavailable: %s", exc)
            return SpotlightResult(
                mention=mention,
                note=f"DBpedia Spotlight is unavailable ({exc}); use search_resource instead",
            )

        matched = _matching_surface_forms(rows, mention)
        candidates = [SpotlightCandidate(**row) for row in matched[: settings.max_results]]
        return SpotlightResult(
            mention=mention,
            candidates=candidates,
            note="" if candidates else f"Spotlight found no link for {mention!r} in this context",
        )

    @mcp.tool()
    def search_resource(
        label: str, expected_types: list[str] | None = None
    ) -> ResourceSearchResult:
        """
        Search DBpedia resources by name.

        Use this when you have no usable context for a mention, or to
        cross-check what `spotlight_link` proposed.

        `expected_types` narrows the search to a class and sharply improves
        precision — pass `["dbo:City"]` when you know you are after a city.
        Any spelling works (`City`, `dbo:City`, the full ontology URI). Use
        `search_class` first if you do not know the class name. If the type
        filter matches nothing, the search is retried unfiltered rather than
        returning empty.

        Results are ranked by the search engine, and rank is not truth:
        `ref_count` is a popularity prior, so a famous wrong answer can outrank
        the right one. Confirm with `get_resource_profile` before committing.
        """
        logger.info("search_resource(%r, %s)", label, expected_types)
        settings = get_settings()

        term = (label or "").strip()
        if not term:
            return ResourceSearchResult(label=label, note="no label given")

        types = normalize_classes(expected_types)
        notes: list[str] = []

        rows: list[dict[str, object]] = []
        try:
            # Lookup accepts a single typeName, so query once per type (capped
            # — two filtered queries plus a fallback is already three requests).
            for type_uri in types[:2]:
                rows += lookup.search(
                    term,
                    type_name=short_name(type_uri),
                    max_results=settings.max_results,
                    settings=settings,
                )
            if not rows:
                if types:
                    notes.append("no match under the requested types; searched unfiltered")
                rows = lookup.search(term, max_results=settings.max_results, settings=settings)
        except lookup.LookupError_ as exc:
            logger.warning("lookup unavailable: %s", exc)
            notes.append(f"DBpedia Lookup is unavailable ({exc})")

        hits = [ResourceHit(source="lookup", **row) for row in rows]  # type: ignore[arg-type]

        # Exact-label SPARQL match as a second source. Lookup is fuzzy and
        # popularity-weighted, so an obscure resource whose label matches
        # exactly can be missing from it entirely.
        try:
            hits += _exact_label_resources(term, types, settings.max_results)
        except sparql.SPARQLError as exc:
            logger.warning("exact-label query failed: %s", exc)
            notes.append(f"exact-label search failed ({exc})")

        merged = _dedupe_by_uri(hits)[: settings.max_results]
        if not merged:
            notes.append(f"no DBpedia resource found for {term!r}")

        return ResourceSearchResult(
            label=term,
            expected_types=types,
            results=merged,
            note="; ".join(notes),
        )

    @mcp.tool()
    def get_resource_profile(uri: str) -> ResourceProfile:
        """
        Fetch what DBpedia knows about one resource, to confirm or reject it.

        This is the confirmation step. Before committing to a candidate, read
        its abstract and types and check they match the entity's evidence in
        the source documents — this is how you catch the city that is really a
        band, or a person who died two centuries before the article.

        Two fields override everything else:

        - `redirects_to` — this URI is a redirect. Ground to the target instead.
        - `is_disambiguation` — this is a disambiguation page and is never a
          valid grounding. Search again with more context.

        `found: false` means DBpedia has no triples for the URI at all, which
        usually means the URI was invented rather than returned by a tool.
        """
        logger.info("get_resource_profile(%r)", uri)
        settings = get_settings()

        try:
            term = sparql.uri_term(uri)
        except ValueError as exc:
            return ResourceProfile(uri=uri, note=f"rejected URI: {exc}")

        try:
            facts = sparql.run_select(
                f"""
                SELECT ?p ?o WHERE {{
                  VALUES ?p {{ rdfs:label dbo:abstract rdf:type owl:sameAs
                               dbo:wikiPageRedirects dbo:wikiPageDisambiguates }}
                  {term} ?p ?o .
                  FILTER(!isLiteral(?o) || LANG(?o) = "" || LANG(?o) = "en")
                  # Narrowed in the query, not afterwards: a well-known resource
                  # carries hundreds of YAGO/Wikidata rdf:type and owl:sameAs
                  # triples, and they would exhaust the LIMIT before the
                  # abstract — the one field this profile most needs — arrived.
                  FILTER(?p != rdf:type  || STRSTARTS(STR(?o), "{DBO}"))
                  FILTER(?p != owl:sameAs || STRSTARTS(STR(?o), "http://www.wikidata.org/"))
                }} LIMIT 200
                """,
                settings=settings,
            )
        except sparql.SPARQLError as exc:
            return ResourceProfile(uri=uri, note=f"DBpedia SPARQL is unavailable ({exc})")

        if not facts:
            return ResourceProfile(
                uri=uri,
                note="no triples in DBpedia for this URI — it is not a real resource",
            )

        buckets: dict[str, list[str]] = {}
        for row in facts:
            predicate, value = row.get("p", ""), row.get("o", "")
            if predicate and value:
                buckets.setdefault(predicate, []).append(value)

        label = _first(buckets.get(_RDFS_LABEL))
        types = {t for t in buckets.get(_RDF_TYPE, []) if t.startswith(DBO)}

        # The public SPARQL snapshot currently serves no dbo:abstract and no
        # rdfs:comment for resources — only Lookup still carries the text. Since
        # the abstract is the whole point of a profile (it is what tells the city
        # from the band), fall back to Lookup rather than return a profile with
        # a hole in it.
        abstract = _first(buckets.get(f"{DBO}abstract"))
        if not abstract:
            abstract = _abstract_from_lookup(uri, label, settings)

        return ResourceProfile(
            uri=uri,
            found=True,
            label=label,
            abstract=abstract[: settings.abstract_chars],
            # Most specific first is a better read for a model than alphabetical;
            # DBpedia publishes no class depth, so name length is the proxy.
            types=sorted(types, key=len, reverse=True),
            wikidata_uri=_first([u for u in buckets.get(_OWL_SAMEAS, []) if "wikidata.org" in u]),
            redirects_to=_first(buckets.get(f"{DBO}wikiPageRedirects")),
            is_disambiguation=bool(buckets.get(f"{DBO}wikiPageDisambiguates")),
            sample_predicates=_sample_predicates(term, settings),
        )

    @mcp.tool()
    def search_class(label: str) -> ClassSearchResult:
        """
        Find DBpedia ontology classes whose name matches a word.

        Use this to turn a type you have in mind into an argument for
        `search_resource`'s `expected_types`, or for the `subject_types` /
        `object_types` of the relation tools.

        Worth calling rather than guessing: DBpedia's ontology is
        British-spelled (`dbo:Organisation`, not `Organization`) and its names
        are not always the obvious ones — a town is a `dbo:Settlement`, a
        company is a `dbo:Company` under `dbo:Organisation`. The
        `superclasses` field shows you where each class sits.
        """
        logger.info("search_class(%r)", label)
        settings = get_settings()

        term = (label or "").strip()
        if not term:
            return ClassSearchResult(label=label, note="no label given")

        needle = _search_phrase(term)
        try:
            rows = sparql.run_select(
                f"""
                SELECT ?uri (SAMPLE(?label) AS ?label) (SAMPLE(?comment) AS ?comment)
                       (GROUP_CONCAT(DISTINCT STR(?super); separator="|") AS ?supers)
                WHERE {{
                  ?uri a owl:Class ; rdfs:label ?label .
                  FILTER(STRSTARTS(STR(?uri), "{DBO}"))
                  FILTER(LANG(?label) = "en")
                  FILTER(CONTAINS(LCASE(STR(?label)), {sparql.text_literal(needle)}))
                  OPTIONAL {{ ?uri rdfs:comment ?comment . FILTER(LANG(?comment) = "en") }}
                  OPTIONAL {{ ?uri rdfs:subClassOf ?super . FILTER(isIRI(?super)) }}
                }} GROUP BY ?uri LIMIT 50
                """,
                settings=settings,
            )
        except sparql.SPARQLError as exc:
            return ClassSearchResult(label=term, note=f"DBpedia SPARQL is unavailable ({exc})")

        hits = [
            ClassHit(
                uri=row.get("uri", ""),
                label=row.get("label", ""),
                comment=row.get("comment", ""),
                superclasses=[s for s in (row.get("supers") or "").split("|") if s],
            )
            for row in rows
        ]
        hits.sort(key=lambda h: _label_rank(h.label, needle))
        hits = hits[: settings.max_results]

        return ClassSearchResult(
            label=term,
            results=hits,
            note="" if hits else f"no ontology class whose label contains {needle!r}",
        )

    # ══ Relation tools ════════════════════════════════════════════════

    @mcp.tool()
    def find_object_properties(
        relation_text: str,
        subject_types: list[str] | None = None,
        object_types: list[str] | None = None,
    ) -> PropertySearchResult:
        """
        Find ontology properties that link two resources, by relation phrase.

        This is the tool for a relation whose object is another entity
        ("Seattle LOCATED_IN Washington"). For a relation whose object is a
        date, a number, or a string, use `find_datatype_properties` instead.

        `relation_text` is matched against property labels, so give it in
        words: "located in", not "LOCATED_IN" (underscores and case are handled,
        but a phrase that matches no label finds nothing). If the whole phrase
        misses, individual words are tried.

        `subject_types` and `object_types` filter on the property's declared
        domain and range. Properties that declare neither are kept regardless —
        much of DBpedia's ontology leaves them undeclared, and dropping those
        would hide the best matches.

        A property matching by label is a hypothesis, not a fact. Confirm it
        with `get_predicates_between` on the two resources.
        """
        logger.info("find_object_properties(%r)", relation_text)
        return _find_properties(
            relation_text,
            kind="object",
            rdf_class="owl:ObjectProperty",
            domain_types=normalize_classes(subject_types),
            range_types=normalize_classes(object_types),
        )

    @mcp.tool()
    def find_datatype_properties(
        relation_text: str,
        subject_types: list[str] | None = None,
        literal_datatype: str | None = None,
    ) -> PropertySearchResult:
        """
        Find ontology properties whose object is a literal, by relation phrase.

        Use this when the target of the relation is a value rather than an
        entity — a date, a number, a name string. `dbo:foundingDate` and
        `dbo:numberOfEmployees` are datatype properties; `dbo:location` is not.

        `literal_datatype` filters on the property's range: pass `xsd:date`,
        `xsd:integer`, `xsd:string`, or the bare name (`date`, `integer`).
        `subject_types` filters on the domain, and properties that declare no
        domain are kept regardless.
        """
        logger.info("find_datatype_properties(%r, %r)", relation_text, literal_datatype)
        datatype = normalize_datatype(literal_datatype)
        return _find_properties(
            relation_text,
            kind="datatype",
            rdf_class="owl:DatatypeProperty",
            domain_types=normalize_classes(subject_types),
            range_types=[datatype] if datatype else [],
        )

    @mcp.tool()
    def get_property_profile(property_uri: str) -> PropertyProfile:
        """
        Fetch what DBpedia declares about one ontology property.

        Use it to check a candidate from `find_object_properties` before
        committing: the comment says what the property actually means, and
        domain/range say what it is allowed to connect.

        Read `usage_count` too. DBpedia's ontology contains properties that are
        declared but barely used; a count of zero means the property is real but
        would ground the relation to something nothing else in the graph uses.
        """
        logger.info("get_property_profile(%r)", property_uri)
        settings = get_settings()

        try:
            term = sparql.uri_term(property_uri)
        except ValueError as exc:
            return PropertyProfile(uri=property_uri, note=f"rejected URI: {exc}")

        try:
            facts = sparql.run_select(
                f"""
                SELECT ?p ?o WHERE {{
                  VALUES ?p {{ rdf:type rdfs:label rdfs:comment rdfs:domain rdfs:range
                               rdfs:subPropertyOf owl:equivalentProperty }}
                  {term} ?p ?o .
                  FILTER(!isLiteral(?o) || LANG(?o) = "" || LANG(?o) = "en")
                }} LIMIT 100
                """,
                settings=settings,
            )
        except sparql.SPARQLError as exc:
            return PropertyProfile(uri=property_uri, note=f"DBpedia SPARQL is unavailable ({exc})")

        if not facts:
            return PropertyProfile(
                uri=property_uri,
                note="no triples in DBpedia for this URI — it is not a real property",
            )

        buckets: dict[str, list[str]] = {}
        for row in facts:
            predicate, value = row.get("p", ""), row.get("o", "")
            if predicate and value:
                buckets.setdefault(predicate, []).append(value)

        declared = buckets.get(_RDF_TYPE, [])
        kind = ""
        if f"{_OWL}ObjectProperty" in declared:
            kind = "object"
        elif f"{_OWL}DatatypeProperty" in declared:
            kind = "datatype"

        usage, usage_note = _usage_count(term, settings)
        return PropertyProfile(
            uri=property_uri,
            found=True,
            label=_first(buckets.get(_RDFS_LABEL)),
            comment=_first(buckets.get(_RDFS_COMMENT)),
            domain=_first(buckets.get(_RDFS_DOMAIN)),
            range=_first(buckets.get(_RDFS_RANGE)),
            kind=kind,
            sub_property_of=sorted(set(buckets.get(_RDFS_SUBPROPERTY, []))),
            equivalent_properties=sorted(set(buckets.get(f"{_OWL}equivalentProperty", []))),
            usage_count=usage,
            note=usage_note,
        )

    @mcp.tool()
    def get_predicates_between(subject_uri: str, object_uri: str) -> PredicatesBetweenResult:
        """
        List the predicates DBpedia actually asserts between two resources.

        The strongest evidence available for a relation: DBpedia either states
        this edge or it does not. Use it to confirm a property you found by
        label search, and to catch inverted relations — a `reverse` direction
        means the triple exists the other way round, so the local relation's
        subject and object are swapped relative to DBpedia's.

        An empty result is informative, not a failure. It means the two
        resources are not directly linked in DBpedia, which is common for facts
        drawn from recent news. Ground the relation to a property found by
        label instead, and lower your confidence.
        """
        logger.info("get_predicates_between(%r, %r)", subject_uri, object_uri)
        settings = get_settings()

        try:
            subject = sparql.uri_term(subject_uri)
            obj = sparql.uri_term(object_uri)
        except ValueError as exc:
            return PredicatesBetweenResult(
                subject_uri=subject_uri, object_uri=object_uri, note=f"rejected URI: {exc}"
            )

        try:
            rows = sparql.run_select(
                f"""
                SELECT DISTINCT ?p ?direction WHERE {{
                  {{ {subject} ?p {obj} . BIND("forward" AS ?direction) }}
                  UNION
                  {{ {obj} ?p {subject} . BIND("reverse" AS ?direction) }}
                }} LIMIT 50
                """,
                settings=settings,
            )
        except sparql.SPARQLError as exc:
            return PredicatesBetweenResult(
                subject_uri=subject_uri,
                object_uri=object_uri,
                note=f"DBpedia SPARQL is unavailable ({exc})",
            )

        links = [
            PredicateLink(
                predicate=row["p"],
                label=_humanize(row["p"]),
                direction=row.get("direction", "forward"),
            )
            for row in rows
            if row.get("p") and row["p"] not in _NOISE_PREDICATES
        ]
        # Ontology predicates first: dbp: properties are raw infobox scrapes and
        # are far noisier than the curated dbo: ones.
        links.sort(key=lambda link: (not link.predicate.startswith(DBO), link.predicate))

        return PredicatesBetweenResult(
            subject_uri=subject_uri,
            object_uri=object_uri,
            predicates=links,
            note="" if links else "DBpedia asserts no direct edge between these two resources",
        )

    return mcp


# ── Shared helpers ────────────────────────────────────────────────────

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_OWL = "http://www.w3.org/2002/07/owl#"

_RDF_TYPE = f"{_RDF}type"
_RDFS_LABEL = f"{_RDFS}label"
_RDFS_COMMENT = f"{_RDFS}comment"
_RDFS_DOMAIN = f"{_RDFS}domain"
_RDFS_RANGE = f"{_RDFS}range"
_RDFS_SUBPROPERTY = f"{_RDFS}subPropertyOf"
_OWL_SAMEAS = f"{_OWL}sameAs"


def _find_properties(
    relation_text: str,
    *,
    kind: str,
    rdf_class: str,
    domain_types: list[str],
    range_types: list[str],
) -> PropertySearchResult:
    """
    Body shared by `find_object_properties` and `find_datatype_properties`.

    Domain/range filtering happens in Python rather than in SPARQL so that
    "declares no domain" can count as a pass. Expressing that in the query
    would mean an OPTIONAL plus a `!BOUND` filter per constraint, and would
    interact badly with the LIMIT — good matches would be dropped before the
    filter ever saw them.
    """
    settings = get_settings()

    phrase = _search_phrase(relation_text)
    if not phrase:
        return PropertySearchResult(relation_text=relation_text, note="no relation text given")

    # Over-fetch: the LIMIT applies before the Python-side type filter.
    raw_limit = min(settings.max_results * 6, 120)
    notes: list[str] = []

    try:
        rows = _property_rows(phrase, rdf_class, raw_limit, settings)
        # A multi-word phrase matches few property labels literally — "located
        # in" finds `locatedInArea` and misses `location`. Widening to
        # individual words whenever the phrase is thin is what surfaces the
        # property that is usually the right answer. `_label_rank` keeps the
        # phrase matches ahead of the word matches afterwards.
        thin = len(rows) < 3
        if thin:
            for token in _tokens(phrase):
                rows += _property_rows(token, rdf_class, raw_limit, settings)
            if not rows:
                notes.append(f"no property label contains {phrase!r} or any of its words")
            else:
                notes.append(f"few labels contain {phrase!r}; also matched on individual words")
    except sparql.SPARQLError as exc:
        return PropertySearchResult(
            relation_text=relation_text, note=f"DBpedia SPARQL is unavailable ({exc})"
        )

    hits: list[PropertyHit] = []
    for row in rows:
        domain, range_ = row.get("domain", ""), row.get("range", "")
        if not _passes(domain, domain_types) or not _passes(range_, range_types):
            continue
        hits.append(
            PropertyHit(
                uri=row.get("uri", ""),
                label=row.get("label", ""),
                comment=row.get("comment", ""),
                domain=domain,
                range=range_,
                kind=kind,
            )
        )

    hits = _dedupe_by_uri(hits)
    hits.sort(key=lambda h: _label_rank(h.label, phrase))
    hits = hits[: settings.max_results]

    if not hits:
        notes.append(
            f"no {kind} property matched {phrase!r}"
            + (" under the requested types" if domain_types or range_types else "")
        )

    return PropertySearchResult(relation_text=relation_text, results=hits, note="; ".join(notes))


def _property_rows(
    needle: str, rdf_class: str, limit: int, settings: object
) -> list[dict[str, str]]:
    return sparql.run_select(
        f"""
        SELECT ?uri (SAMPLE(?label) AS ?label) (SAMPLE(?comment) AS ?comment)
               (SAMPLE(?domain) AS ?domain) (SAMPLE(?range) AS ?range)
        WHERE {{
          ?uri a {rdf_class} ; rdfs:label ?label .
          FILTER(STRSTARTS(STR(?uri), "{DBO}"))
          FILTER(LANG(?label) = "en")
          FILTER(CONTAINS(LCASE(STR(?label)), {sparql.text_literal(needle)}))
          OPTIONAL {{ ?uri rdfs:comment ?comment . FILTER(LANG(?comment) = "en") }}
          OPTIONAL {{ ?uri rdfs:domain ?domain }}
          OPTIONAL {{ ?uri rdfs:range ?range }}
        }} GROUP BY ?uri LIMIT {limit}
        """,
        settings=settings,  # type: ignore[arg-type]
    )


def _passes(declared: str, wanted: list[str]) -> bool:
    """A property passes a type constraint if it matches it, or declares nothing."""
    return not wanted or not declared or declared in wanted


def _exact_label_resources(label: str, types: list[str], limit: int) -> list[ResourceHit]:
    """Resources whose English label matches `label` exactly, case-insensitively."""
    type_clause = ""
    if types:
        values = " ".join(sparql.uri_term(t) for t in types)
        type_clause = f"VALUES ?type {{ {values} }} ?uri rdf:type ?type ."

    rows = sparql.run_select(
        f"""
        SELECT DISTINCT ?uri ?label WHERE {{
          ?uri rdfs:label ?label .
          {type_clause}
          FILTER(LANG(?label) = "en")
          FILTER(LCASE(STR(?label)) = {sparql.text_literal(label.lower())})
          FILTER(STRSTARTS(STR(?uri), "{DBR}"))
          FILTER NOT EXISTS {{ ?uri dbo:wikiPageDisambiguates ?d }}
          FILTER NOT EXISTS {{ ?uri dbo:wikiPageRedirects ?r }}
        }} LIMIT {limit}
        """
    )
    return [
        ResourceHit(uri=row["uri"], label=row.get("label", ""), source="sparql")
        for row in rows
        if row.get("uri")
    ]


def _abstract_from_lookup(uri: str, label: str, settings: object) -> str:
    """
    The abstract for one resource, via DBpedia Lookup.

    Lookup has no by-URI endpoint, so this searches its label and keeps the row
    whose URI matches exactly — anything else would put a different resource's
    abstract on this profile, which is worse than having none.
    """
    query = label or short_name(uri).replace("_", " ")
    if not query:
        return ""
    try:
        rows = lookup.search(query, max_results=10, settings=settings)  # type: ignore[arg-type]
    except lookup.LookupError_ as exc:
        logger.warning("abstract lookup failed for %s: %s", uri, exc)
        return ""
    for row in rows:
        if row.get("uri") == uri:
            return str(row.get("comment", ""))
    return ""


def _sample_predicates(term: str, settings: object) -> list[PredicateUsage]:
    """A handful of outgoing dbo: predicates, to show what kind of thing this is."""
    try:
        rows = sparql.run_select(
            f"""
            SELECT ?p (SAMPLE(?o) AS ?o) WHERE {{
              {term} ?p ?o .
              FILTER(STRSTARTS(STR(?p), "{DBO}"))
            }} GROUP BY ?p LIMIT 40
            """,
            settings=settings,  # type: ignore[arg-type]
        )
    except sparql.SPARQLError as exc:
        logger.warning("sample-predicate query failed: %s", exc)
        return []

    usages = [
        PredicateUsage(
            predicate=row["p"],
            label=_humanize(row["p"]),
            example_value=(row.get("o") or "")[:120],
        )
        for row in rows
        if row.get("p") and row["p"] not in _NOISE_PREDICATES
    ]
    return usages[:15]


def _usage_count(term: str, settings: object) -> tuple[int, str]:
    """
    How many triples use this property, capped.

    The cap is a nested LIMIT rather than a plain COUNT: some DBpedia
    properties appear in millions of triples and an uncapped count times out.
    """
    try:
        rows = sparql.run_select(
            f"""
            SELECT (COUNT(*) AS ?n) WHERE {{
              SELECT ?s WHERE {{ ?s {term} ?o }} LIMIT 10000
            }}
            """,
            settings=settings,  # type: ignore[arg-type]
        )
    except sparql.SPARQLError as exc:
        return 0, f"usage count unavailable ({exc})"

    try:
        count = int(rows[0]["n"]) if rows else 0
    except (KeyError, ValueError):
        return 0, "usage count unavailable"
    return count, "usage count is capped at 10000" if count >= 10000 else ""


def _matching_surface_forms(rows: list[dict[str, object]], mention: str) -> list[dict[str, object]]:
    """
    Keep the annotations whose surface form corresponds to `mention`.

    Spotlight annotates the whole context, so most rows are about other
    entities in the sentence. Exact matches come first; a containment match is
    accepted as a fallback so "Obama" still matches an annotation of "Barack
    Obama". If nothing matches at all, everything is returned — a caller that
    passed a mention Spotlight spelled differently is better served by the
    sentence's annotations than by an empty list.
    """
    needle = (mention or "").strip().lower()
    if not needle:
        return rows

    exact = [r for r in rows if str(r.get("surface_form", "")).lower() == needle]
    if exact:
        return exact

    partial = [
        r
        for r in rows
        if needle in str(r.get("surface_form", "")).lower()
        or str(r.get("surface_form", "")).lower() in needle
    ]
    return partial or rows


def _dedupe_by_uri[T](items: list[T]) -> list[T]:
    """Keep the first occurrence of each URI, preserving order."""
    seen: set[str] = set()
    out: list[T] = []
    for item in items:
        uri = getattr(item, "uri", "")
        if not uri or uri in seen:
            continue
        seen.add(uri)
        out.append(item)
    return out


def _search_phrase(text: str) -> str:
    """`LOCATED_IN` → `located in`. What a label search can actually match."""
    return re.sub(r"[_\-]+", " ", (text or "").strip().lower()).strip()


def _tokens(phrase: str) -> list[str]:
    """
    Words worth widening a thin phrase search with, longest first.

    Each word is crudely stemmed, because property labels are nouns while
    relation phrases are verbs: the local `LOCATED_IN` has to reach
    `dbo:location`, and only the shared prefix `locat` gets there. Stemming
    to a prefix is safe here — the SPARQL side matches on CONTAINS, so a
    shorter needle only ever widens the net, and `_label_rank` re-tightens it
    by putting the closest labels first.
    """
    stems: list[str] = []
    for word in phrase.split():
        if len(word) <= 3:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        stems.append(word)
    return sorted(dict.fromkeys(stems), key=len, reverse=True)[:3]


def _label_rank(label: str, needle: str) -> tuple[int, int, str]:
    """Sort key: exact match, then prefix match, then shortest label."""
    lowered = (label or "").lower()
    if lowered == needle:
        return (0, len(lowered), lowered)
    if lowered.startswith(needle):
        return (1, len(lowered), lowered)
    return (2, len(lowered), lowered)


def _humanize(uri: str) -> str:
    """`http://dbpedia.org/ontology/birthPlace` → `birth place`."""
    return _CAMEL.sub(" ", short_name(uri)).replace("_", " ").lower()


def _first(values: list[str] | None) -> str:
    return values[0] if values else ""
