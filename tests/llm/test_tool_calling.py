"""
`LangChainToolLoopMixin` — the loop shared by all three provider adapters.

The chat model is a fake `BaseChatModel`-shaped object, because what is being
tested is the loop's own bookkeeping: that tool calls are executed and fed back,
that the budget is respected, and that the final call is made with tools
*unbound*. None of that needs a provider.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMStructuredOutputError, ToolInvocation, ToolSpec
from kg_agentic_extraction.llm.tool_calling import LangChainToolLoopMixin


class Answer(BaseModel):
    text: str = ""


class FakeStructuredRunnable:
    def __init__(self, owner: FakeChatModel) -> None:
        self._owner = owner

    def invoke(self, messages: list[Any]) -> Answer:
        self._owner.final_transcripts.append(messages)
        if self._owner.final_raises is not None:
            raise self._owner.final_raises
        return Answer(text="done")


class FakeBoundModel:
    def __init__(self, owner: FakeChatModel) -> None:
        self._owner = owner

    def invoke(self, messages: list[Any]) -> AIMessage:
        self._owner.invocations.append(list(messages))
        if self._owner.invoke_raises is not None:
            raise self._owner.invoke_raises
        turn = self._owner.turns.pop(0) if self._owner.turns else []
        return AIMessage(content="", tool_calls=list(turn))


class FakeChatModel:
    """The surface `LangChainToolLoopMixin` uses: `bind_tools` and `with_structured_output`."""

    def __init__(self, turns: list[list[dict]]) -> None:
        self.turns = turns
        self.bound_tools: list[dict] | None = None
        self.invocations: list[list[Any]] = []
        self.final_transcripts: list[list[Any]] = []
        self.structured_calls = 0
        self.invoke_raises: Exception | None = None
        self.final_raises: Exception | None = None

    def bind_tools(self, tools: list[dict]) -> FakeBoundModel:
        self.bound_tools = tools
        return FakeBoundModel(self)

    def with_structured_output(self, schema, **kwargs) -> FakeStructuredRunnable:  # noqa: ARG002
        self.structured_calls += 1
        return FakeStructuredRunnable(self)


class Client(LangChainToolLoopMixin):
    """A minimal adapter, shaped like the three real ones."""

    def __init__(self, llm: FakeChatModel) -> None:
        self._llm = llm
        self._model_name = "fake"

    def structured(self, *, system: str, user: str, schema):  # noqa: ARG002
        self._llm.structured_calls += 1
        return schema()


TOOLS = [ToolSpec(name="search_class", description="find a class", input_schema={"type": "object"})]


def call(name: str, args: dict, call_id: str = "c0") -> dict:
    return {"name": name, "args": args, "id": call_id}


def run(client: Client, executed: list[ToolInvocation], **kwargs) -> Answer:
    def execute(invocation: ToolInvocation) -> str:
        executed.append(invocation)
        return f'{{"ok": "{invocation.name}"}}'

    return client.run_tool_loop(
        system="s", user="u", tools=TOOLS, execute=execute, schema=Answer, **kwargs
    )


# ── The happy path ────────────────────────────────────────────────────


def test_tool_calls_are_executed_and_fed_back():
    llm = FakeChatModel([[call("search_class", {"label": "City"})], []])
    executed: list[ToolInvocation] = []

    assert run(Client(llm), executed).text == "done"

    assert [i.name for i in executed] == ["search_class"]
    assert executed[0].arguments == {"label": "City"}
    # The result must reach the model as a ToolMessage on the next turn.
    second_turn = llm.invocations[1]
    assert isinstance(second_turn[-1], ToolMessage)
    assert second_turn[-1].content == '{"ok": "search_class"}'


def test_the_loop_stops_when_the_model_asks_for_no_more_tools():
    llm = FakeChatModel([[], [call("search_class", {})]])
    executed: list[ToolInvocation] = []
    run(Client(llm), executed)
    assert executed == []
    assert len(llm.invocations) == 1


def test_parallel_calls_in_one_turn_are_all_executed():
    llm = FakeChatModel(
        [[call("search_class", {"label": "A"}, "a"), call("search_class", {"label": "B"}, "b")], []]
    )
    executed: list[ToolInvocation] = []
    run(Client(llm), executed)
    assert [i.arguments["label"] for i in executed] == ["A", "B"]
    assert [i.id for i in executed] == ["a", "b"]


def test_tools_are_advertised_in_the_openai_function_shape():
    llm = FakeChatModel([[]])
    run(Client(llm), [])
    assert llm.bound_tools == [
        {
            "type": "function",
            "function": {
                "name": "search_class",
                "description": "find a class",
                "parameters": {"type": "object"},
            },
        }
    ]


# ── The final call ────────────────────────────────────────────────────


def test_the_final_call_is_made_with_tools_unbound():
    """Structured output and tool binding are two competing constraints on one
    generation; the closing call must carry only the first."""
    llm = FakeChatModel([[call("search_class", {})], []])
    run(Client(llm), [])

    assert llm.structured_calls == 1
    # It goes through with_structured_output on the raw model, never on the
    # bound one — the bound model only ever produced AIMessages.
    assert all(not isinstance(m, Answer) for m in llm.invocations[-1])
    assert llm.final_transcripts, "the transcript must reach the closing call"


def test_the_final_call_sees_the_whole_transcript():
    llm = FakeChatModel([[call("search_class", {})], []])
    run(Client(llm), [])

    transcript = llm.final_transcripts[0]
    assert isinstance(transcript[0], SystemMessage)
    assert any(isinstance(m, ToolMessage) for m in transcript)


def test_a_failed_final_call_raises_a_typed_error():
    llm = FakeChatModel([[]])
    llm.final_raises = RuntimeError("provider said no")
    with pytest.raises(LLMStructuredOutputError):
        run(Client(llm), [])


# ── Degradation ───────────────────────────────────────────────────────


def test_no_tools_falls_back_to_a_direct_call():
    llm = FakeChatModel([])
    client = Client(llm)
    result = client.run_tool_loop(
        system="s", user="u", tools=[], execute=lambda i: "", schema=Answer
    )
    assert isinstance(result, Answer)
    assert llm.bound_tools is None


def test_spending_the_budget_still_produces_an_answer():
    llm = FakeChatModel([[call("search_class", {"i": str(i)})] for i in range(10)])
    executed: list[ToolInvocation] = []

    assert run(Client(llm), executed, max_rounds=2).text == "done"
    assert len(executed) == 2


def test_a_provider_failure_mid_loop_salvages_the_transcript():
    """Several round-trips have already been paid for; losing them to answer
    nothing is strictly worse than answering with what was gathered."""
    llm = FakeChatModel([[call("search_class", {})], []])
    executed: list[ToolInvocation] = []

    def execute(invocation: ToolInvocation) -> str:
        executed.append(invocation)
        llm.invoke_raises = RuntimeError("rate limited")
        return "{}"

    result = Client(llm).run_tool_loop(
        system="s", user="u", tools=TOOLS, execute=execute, schema=Answer
    )
    assert result.text == "done"
    assert len(executed) == 1


def test_an_exception_from_execute_becomes_a_readable_tool_result():
    llm = FakeChatModel([[call("search_class", {})], []])

    def boom(invocation: ToolInvocation) -> str:
        raise RuntimeError("backend exploded")

    Client(llm).run_tool_loop(system="s", user="u", tools=TOOLS, execute=boom, schema=Answer)

    tool_message = llm.invocations[1][-1]
    assert isinstance(tool_message, ToolMessage)
    assert "backend exploded" in tool_message.content


def test_a_tool_call_with_a_null_id_still_gets_one():
    """LangChain types the id as optional and some providers leave it unset, but
    ToolMessage requires one to correlate the result."""
    llm = FakeChatModel([[{"name": "search_class", "args": {}, "id": None}], []])
    executed: list[ToolInvocation] = []
    run(Client(llm), executed)

    assert executed[0].id
    assert llm.invocations[1][-1].tool_call_id == executed[0].id


def test_a_tool_call_with_no_name_is_skipped():
    llm = FakeChatModel([[{"name": "", "args": {}, "id": "x"}], []])
    executed: list[ToolInvocation] = []
    run(Client(llm), executed)
    assert executed == []
