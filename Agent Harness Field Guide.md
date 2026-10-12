Claude Agent SDK · Python · Field guide

# Building a Claude agent harness, one block at a time

Each of the eight blocks below gets its internal flow, its parts, working code, and the mistakes people make with it. Every block follows the same request through the same agent, so the code pieces fit together at the end.

Running exampleA support agent for an online shop. A customer writes **"Where's my refund for order 1182?"** The agent looks up the order, issues refunds up to $100 on its own, and hands anything larger to a human.

## The whole harness

Select any block to jump to it. The coral block is the model; everything in blue is code you write.

Blocks

1. [01Trigger](#trigger)
2. [02Context](#context)
3. [03Claude](#claude)
4. [04Tools](#tools)
5. [05Memory](#memory)
6. [06Guardrails](#guardrails)
7. [07Observability](#observability)
8. [08Result](#result)
9. [··All together](#assembled)
10. [··What you'll learn](#learn)

01 · Entry point

## Trigger and interface

This is where work enters the agent. It answers three questions before Claude sees anything: who is asking, through which channel, and is this a new conversation or a continuing one.

*1*Event arrives→ *2*Authenticate→ *3*Find or start session→ *4*Run the agent→ *5*Save session id

### What's inside

Channels

Chat widget, REST endpoint, Slack, email, a ticket queue. Each one is an adapter that turns its event into a prompt.

Identity

Your existing auth. The agent should never be the thing deciding who a user is.

Sessions

The SDK gives every run a `session_id`. Store it per user and pass it back as `resume` to continue the conversation.

Scheduled runs

A cron job or queue worker that calls the same function with a fixed prompt, with no human on the other end.

Refund agentThe help widget posts the customer's message to `/support/chat`. The harness finds this customer's last session and resumes it, so "it" in "where's it now?" still means order 1182.

Common mistakes

- Passing the user id inside the prompt and trusting Claude to keep it. Bind it in code instead (block 05).
- Starting a fresh session on every message, which loses the thread.
- Blocking the HTTP request on long tasks. Queue them and notify when done.

**app.py**

```
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from claude_agent_sdk import query, ResultMessage

from harness import build_context, build_options   # blocks 02-06
from auth import current_user                         # your existing auth
from store import sessions                            # e.g. Redis: user_id -> session_id

app = FastAPI()

class ChatIn(BaseModel):
    text: str

@app.post("/support/chat")
async def chat(body: ChatIn, user=Depends(current_user)):
    prior = sessions.get(user.id)                     # None on first contact
    options = build_options(user, resume=prior)
    prompt = build_context(user, body.text)

    reply = None
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, ResultMessage):            # last message of every run
            sessions.set(user.id, msg.session_id)
            reply = msg.result
    return {"reply": reply}
```

- L17`resume=prior` makes this a continuing conversation. Leave it as `None` to start fresh.
- L21The SDK streams messages as the agent works. `ResultMessage` always comes last and carries the final text, the session id, and cost.
- TipFor a chat UI that needs interrupts or many back-and-forth turns in one connection, use `ClaudeSDKClient` instead of `query()`.

02 · What Claude sees

## Context assembly

Claude only knows what is in front of it on each turn. This block decides exactly that: the standing instructions, the facts relevant to this request, and nothing that would distract.

*1*System prompt→ *2*Retrieve facts→ *3*Add memories→ *4*Tag and order→ *5*Compact when long

### What's inside

System prompt

Role, scope, rules, tone and output length. It stays the same for every user, which also makes it cache well.

Retrieved facts

Only what this request needs: the refund policy section and this customer's recent orders, not the whole knowledge base.

XML-style tags

Wrapping each source in its own tag helps Claude tell policy from data from the customer's words.

Compaction

The SDK summarizes older turns automatically when the context fills up. A `PreCompact` hook lets you save notes first.

Refund agentThe prompt includes two policy paragraphs found by search, the last three orders from the database, and one remembered fact: "prefers email updates."

Common mistakes

- Pasting entire documents "just in case." More context means slower, costlier and less focused answers.
- Rules that conflict ("be brief" and "explain fully").
- Putting per-user data in the system prompt, which breaks caching.

**harness/context.py**

```
from kb import search          # your vector store or keyword search
from db import recent_orders
from memory import facts_for   # block 05

SYSTEM = """You are Sami, the support agent for Northwind Shop.
Scope: orders, shipping and refunds only. Politely decline anything else.
Rules:
- Call get_order before saying anything about an order.
- Follow the refund policy exactly. If a case is unclear, call escalate.
- Never invent order numbers, dates or amounts.
Style: 2-4 short sentences, friendly and plain."""

def build_context(user, text: str) -> str:
    policy = search("refund policy " + text, k=2)
    orders = recent_orders(user.id, limit=3)
    facts = facts_for(user.id)

    return f"""<customer id="{user.id}" tier="{user.tier}">
{facts}
</customer>
<recent_orders>
{orders}
</recent_orders>
<policy>
{policy}
</policy>

Customer message:
{text}"""
```

- L5The system prompt names scope, rules and style separately. Short, specific rules beat long paragraphs.
- L14Retrieval runs per request, using the customer's own words as the search query.
- TipOn the raw Messages API, mark the stable system prompt with `cache_control: {"type": "ephemeral"}` so repeat calls are cheaper and faster. The Agent SDK handles prompt caching for you.

03 · The model

## Claude and the agent loop

On each turn Claude reads the context and either answers or asks for a tool. The loop keeps going until Claude answers without asking for a tool, or a limit stops it. The Agent SDK runs this loop for you; below is what it does underneath.

*1*Read context→ *2*Think→ *3*Return tool_use or text→ *4*Harness runs tool→ ↻ back to 1 until stop_reason is end_turn

### What's inside

stop_reason

`tool_use` means "run these tools and send me the results." `end_turn` means Claude is done.

Thinking and effort

Set `effort` (low to max) to trade speed for depth. Hard policy calls benefit from more; simple lookups don't.

Model choice

Use a larger model for judgment and a smaller, faster one for routing or simple subagents. Subagents can each set their own model.

Turn limit

Always cap the loop. A confused agent can otherwise call tools forever.

Refund agentTurn 1: Claude returns `get_order(1182)`. Turn 2: it sees "refund pending, $42" and returns `issue_refund`. Turn 3: it writes the reply and stops.

Common mistakes

- Forgetting to append Claude's own message before the tool results, which breaks the conversation.
- Not matching each `tool_result` to its `tool_use_id`.
- No turn limit.

**loop_from_scratch.py · what the SDK does for you**

```
import json
import anthropic

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY

def run_agent(user_text, system, tools, execute, max_turns=8):
    messages = [{"role": "user", "content": user_text}]

    for _ in range(max_turns):
        resp = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=2048,
            system=system,
            tools=tools,                  # [{"name", "description", "input_schema"}]
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":          # end_turn: Claude is done
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                output = execute(block.name, block.input)   # your code runs it
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output),
                })
        messages.append({"role": "user", "content": results})

    return "Stopped: turn limit reached."
```

- L17Claude's reply, including its `tool_use` blocks, goes back into the history before the results.
- L22Claude can ask for several tools in one turn. Run them all and return every result in a single user message.
- TipWith the Agent SDK you skip all of this and set `model`, `effort` and `max_turns` on `ClaudeAgentOptions` (see the final section).

04 · Acting in the world

## Tools via MCP

Claude only asks for a tool. Your code decides whether to run it and what comes back. Good tools have plain names, descriptions that say when to use them, and error messages that tell Claude what to do next.

*1*Claude requests tool→ *2*Validate input→ *3*Guardrail check→ *4*Execute→ *5*Return result

### What's inside

@tool

Name, description and input schema, plus an async function that does the work and returns content blocks.

In-process MCP server

`create_sdk_mcp_server` bundles your tools. Claude sees them as `mcp__support__get_order` and so on.

External MCP servers

Connect a CRM or database that already speaks MCP over HTTP, with no wrapper code.

Annotations

Hints such as `readOnlyHint` tell clients which tools only read and which change things.

Refund agentThree tools: `get_order` (read only), `issue_refund` (moves money, gated in block 06) and `escalate` (opens a ticket for a human).

Common mistakes

- Vague descriptions like "order tool." Claude picks tools by their description.
- Returning a stack trace. Return a sentence Claude can act on.
- One giant tool with a `mode` parameter instead of small, clear tools.

**harness/tools.py**

```
import json
from typing import Any
from claude_agent_sdk import tool, create_sdk_mcp_server, ToolAnnotations

import db, payments, tickets
from harness.memory import CURRENT_USER, remember   # block 05

def reply(data: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(data)}]}

@tool("get_order",
      "Look up one order by its ID. Call this before discussing any order.",
      {"order_id": str},
      annotations=ToolAnnotations(readOnlyHint=True))
async def get_order(args):
    order = await db.get_order(args["order_id"], owner=CURRENT_USER.get())
    if order is None:
        return reply({"error": "No such order for this customer. "
                               "Ask them to check the order number."})
    return reply(order)

@tool("issue_refund",
      "Refund an order in USD. Only when the refund policy allows it.",
      {"order_id": str, "amount": float, "reason": str})
async def issue_refund(args):
    ref = await payments.refund(args["order_id"], args["amount"], args["reason"])
    return reply({"refund_id": ref.id, "eta_days": "3-5"})

@tool("escalate",
      "Hand the case to a human teammate. Include a 2-sentence summary.",
      {"summary": str})
async def escalate(args):
    t = await tickets.open(CURRENT_USER.get(), args["summary"])
    return reply({"ticket": t.id, "response_time": "within 4 hours"})

support_server = create_sdk_mcp_server(
    name="support", version="1.0.0",
    tools=[get_order, issue_refund, escalate, remember],
)

SUPPORT_TOOLS = [
    "mcp__support__get_order", "mcp__support__issue_refund",
    "mcp__support__escalate", "mcp__support__remember",
]

# An existing MCP server elsewhere can be added the same way:
# mcp_servers={"crm": {"type": "http", "url": "https://crm.example.com/mcp"}}
```

- L16The owner check happens in code. Even if Claude is tricked into asking for someone else's order, the database won't return it.
- L18The error explains what to do next, so Claude can recover instead of guessing.
- L41Tool names follow the pattern `mcp__<server>__<tool>`. You will use these names in permissions and hooks.

05 · What persists

## Memory

Agents have three kinds of memory: the current context window, the session (resumable conversation), and long-term facts that outlive any session. Each needs a different home.

*1*Load facts at start→ *2*Resume session→ *3*Claude saves new facts→ *4*Save notes before compaction

### What's inside

Working memory

The context window itself. Fast, but it is reset or compacted.

Session

Stored transcript you continue with `resume`. Use `session_store` to keep transcripts in your own backend across servers.

Long-term facts

Your own table of short facts per user, written by a `remember` tool and read back in block 02.

Bound identity

A context variable holds the current user, so tools never take a user id from Claude.

Refund agentClaude calls `remember("Prefers email updates; had a damaged-item refund in March")`. Next month that fact is in the context before the customer says a word.

Common mistakes

- Storing whole transcripts as "memory." Store short, useful facts.
- No way for users to see or delete what is remembered.
- Letting one customer's memory leak into another's context.

**harness/memory.py**

```
import json
from contextvars import ContextVar
from claude_agent_sdk import tool
import db

CURRENT_USER: ContextVar[str] = ContextVar("current_user")

def facts_for(user_id: str) -> str:
    rows = db.memory.list(user_id, limit=10)      # newest first
    return "\n".join(f"- {r.fact}" for r in rows) or "- (no saved facts)"

@tool("remember",
      "Save one durable fact about this customer, such as a preference "
      "or a past issue. One short sentence. Never save payment details.",
      {"fact": str})
async def remember(args):
    fact = args["fact"].strip()[:200]
    await db.memory.add(CURRENT_USER.get(), fact)
    return {"content": [{"type": "text", "text": json.dumps({"saved": True})}]}

async def save_before_compact(input_data, tool_use_id, context):
    # Fires before the SDK summarizes old turns (PreCompact hook).
    await db.notes.add(input_data["session_id"], "context compacted")
    return {}
```

- L6Set `CURRENT_USER` at the start of each request (see the final section). Tools read it; Claude never supplies it.
- L12The description sets the rules for what counts as memory, including what never to store.
- L21Register this under `"PreCompact"` in the hooks to keep a record when old turns get summarized.

06 · Safety and policy

## Guardrails

Prompts ask Claude to behave; guardrails make sure it does. They run in your code, before tools execute, so they hold even if a prompt is ignored or manipulated.

*1*Tool requested→ *2*Is it allowed at all?→ *3*PreToolUse hook checks policy→ *4*Allow or deny with reason

### What's inside

Tool allowlist

`tools=[]` removes built-in shell and file tools. `allowed_tools` plus `permission_mode="dontAsk"` denies anything not listed.

PreToolUse hooks

Run before every matching call. Return `deny` with a reason and Claude reads that reason and adapts.

Hard limits

`max_turns` and `max_budget_usd` stop runaway runs.

Human approval

For actions above a threshold, deny and route to a human through `escalate`.

Refund agent$42 passes the hook and runs. A $240 request is denied with "needs a human, call escalate," so Claude opens a ticket and tells the customer.

Common mistakes

- Relying on "never refund more than $100" in the prompt alone.
- Denying without a reason, so Claude retries the same call.
- Using `bypassPermissions` in production.

**harness/guardrails.py**

```
from claude_agent_sdk import HookMatcher
import db, policy

REFUND_LIMIT_USD = 100

def deny(reason: str) -> dict:
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }}

async def refund_guard(input_data, tool_use_id, context):
    args = input_data["tool_input"]
    amount = float(args.get("amount", 0))

    if amount > REFUND_LIMIT_USD:
        return deny(f"Refunds over ${REFUND_LIMIT_USD} need a human. "
                    "Call escalate with a short summary instead.")

    order = await db.get_order(args["order_id"])
    if order is None or not policy.refund_eligible(order, amount):
        return deny("This order is not eligible under the refund policy. "
                    "Explain why to the customer, or escalate if unsure.")
    return {}                                    # no objection: proceed

GUARD_HOOKS = {
    "PreToolUse": [
        HookMatcher(matcher="mcp__support__issue_refund", hooks=[refund_guard]),
    ],
}

LIMITS = dict(
    tools=[],                      # no built-in shell or file tools
    permission_mode="dontAsk",     # anything not in allowed_tools is denied
    max_turns=8,
    max_budget_usd=0.50,
)
```

- L18The deny reason is written for Claude. It says what went wrong and what to do instead.
- L21Re-check eligibility from the database. Don't trust amounts or claims that only appear in the conversation.
- L29The matcher limits this hook to one tool. A hook without a matcher runs for every tool.

07 · Seeing and proving it works

## Observability

You can't improve an agent you can't see. Log every tool call, record cost and turns for every run, and keep a test set that you rerun whenever you change the prompt, tools or model.

*1*Log tool calls→ *2*Record run metrics→ *3*Review failures→ *4*Add to eval set→ ↻ rerun evals after every change

### What's inside

Audit log

A `PostToolUse` hook records tool name, input and output for every call.

Run metrics

`ResultMessage` carries `num_turns`, `duration_ms`, `total_cost_usd` and `is_error`.

Evals

Fixed inputs with expected behavior, run against a test database. Check which tools were called, not exact wording.

Transcripts

The saved session lets you replay exactly what Claude saw when something went wrong.

Refund agentDashboard: 1,240 chats this week, 3.1 turns and $0.04 per chat on average, 6% escalated. One eval case caught a prompt change that made Claude skip `get_order`.

Common mistakes

- Running evals against production systems.
- Grading exact sentences, which fails on harmless rewording.
- Logging card numbers or other secrets from tool inputs.

**harness/observe.py**

```
import logging
from claude_agent_sdk import (HookMatcher, query, AssistantMessage,
                              ToolUseBlock, ResultMessage)

log = logging.getLogger("agent")

async def audit(input_data, tool_use_id, context):
    log.info("tool_call", extra={
        "session": input_data["session_id"],
        "tool": input_data["tool_name"],
        "input": input_data["tool_input"],
        "output": str(input_data.get("tool_response"))[:500],
    })
    return {}

AUDIT_HOOKS = {"PostToolUse": [HookMatcher(hooks=[audit])]}

def record(msg: ResultMessage):
    log.info("run_done", extra={
        "session": msg.session_id, "turns": msg.num_turns,
        "ms": msg.duration_ms, "cost_usd": msg.total_cost_usd,
        "ok": not msg.is_error, "subtype": msg.subtype,
    })

# Evals: run against a TEST database, check behavior not wording
CASES = [
    ("Refund order 1182, it arrived broken ($42)", "mcp__support__issue_refund"),
    ("Refund order 2207, the $240 TV",            "mcp__support__escalate"),
    ("What's your CEO's phone number?",            None),
]

async def tools_used(prompt, options):
    names = []
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            names += [b.name for b in msg.content if isinstance(b, ToolUseBlock)]
    return names

async def run_evals(make_options, make_prompt):
    passed = 0
    for text, expected in CASES:
        used = await tools_used(make_prompt(text), make_options())
        ok = (expected in used) if expected else not any(
            n.endswith(("issue_refund", "escalate")) for n in used)
        passed += ok
        print("PASS" if ok else "FAIL", text, used)
    print(f"{passed}/{len(CASES)} passed")
```

- L7A hook without a matcher runs after every tool call, so nothing is missed.
- L18Call `record()` on each `ResultMessage` in your endpoint. Cost is an estimate computed by the SDK.
- L36Reading `ToolUseBlock`s from the message stream shows exactly which tools Claude chose.

08 · Handing back

## Result delivered

The run ends with a `ResultMessage`. Check whether it succeeded, turn it into something your app can act on, and have a plan for when it didn't.

*1*Check subtype→ *2*Read structured output→ *3*Update your systems→ *4*Reply or hand off

### What's inside

Subtype

`success`, or an error such as `error_max_turns` or `error_max_budget_usd`.

Structured output

Set `output_format` to a JSON schema and read `structured_output`, so your code never parses prose.

Side effects

Close the ticket, update the CRM, notify the customer by their preferred channel.

Fallback

When a run fails, give the user a clear message and route to a human with the session attached.

Refund agentOutput: `{"status": "resolved", "refund_usd": 42, "reply": "Your $42 refund was issued today and will arrive in 3-5 business days."}` The ticket closes automatically.

Common mistakes

- Treating any final text as success.
- Parsing Claude's prose with regex to find the status.
- Showing the customer a raw error.

**harness/result.py**

```
from claude_agent_sdk import ResultMessage
import tickets, notify

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "status": {"type": "string",
                   "enum": ["resolved", "escalated", "needs_info"]},
        "refund_usd": {"type": ["number", "null"]},
    },
    "required": ["reply", "status"],
}

OUTPUT = dict(output_format={"type": "json_schema", "schema": REPLY_SCHEMA})

FALLBACK = {"reply": "I'm passing this to a teammate who will reply "
                     "within 4 hours.", "status": "escalated"}

async def handle(user, msg: ResultMessage) -> dict:
    if msg.subtype == "success" and not msg.is_error and msg.structured_output:
        out = msg.structured_output
        if out["status"] == "resolved":
            await tickets.close_for(user.id, session=msg.session_id)
        await notify.send(user, out["reply"])     # email, if they prefer it
        return out

    # error_max_turns, error_max_budget_usd, API errors ...
    await tickets.open(user.id, f"Agent run failed: {msg.subtype}",
                       session=msg.session_id)
    return FALLBACK
```

- L15With `output_format` set, the SDK validates Claude's final answer against your schema.
- L21Check both `subtype` and `is_error`. A run can finish its loop and still end on a failed API call.
- L30The session id goes with the handoff, so the human sees the full conversation.

All eight blocks

## Putting it together

This is the one function that wires every block into `ClaudeAgentOptions`, plus the updated endpoint. Swap the mock modules (`db`, `payments`, `tickets`) for your own and it runs.

**harness/\_\_init\_\_.py**

```
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from .context import SYSTEM, build_context                 # 02
from .tools import support_server, SUPPORT_TOOLS            # 04
from .memory import CURRENT_USER, save_before_compact       # 05
from .guardrails import GUARD_HOOKS, LIMITS                 # 06
from .observe import AUDIT_HOOKS, record                    # 07
from .result import OUTPUT, handle                          # 08

def build_options(user, resume=None) -> ClaudeAgentOptions:
    CURRENT_USER.set(user.id)                               # bind identity
    return ClaudeAgentOptions(
        system_prompt=SYSTEM,
        model="claude-sonnet-5-5",                          # 03
        effort="medium",
        mcp_servers={"support": support_server},
        allowed_tools=SUPPORT_TOOLS,
        hooks={
            **GUARD_HOOKS,
            **AUDIT_HOOKS,
            "PreCompact": [HookMatcher(hooks=[save_before_compact])],
        },
        resume=resume,
        setting_sources=[],              # ignore local CLAUDE.md and settings
        **LIMITS,
        **OUTPUT,
    )


# app.py, the endpoint from block 01 with blocks 07-08 added
@app.post("/support/chat")
async def chat(body: ChatIn, user=Depends(current_user)):
    options = build_options(user, resume=sessions.get(user.id))
    out = None
    async for msg in query(prompt=build_context(user, body.text), options=options):
        if isinstance(msg, ResultMessage):
            sessions.set(user.id, msg.session_id)
            record(msg)
            out = await handle(user, msg)
    return out
```

Install with `pip install claude-agent-sdk` inside a virtual environment and set `ANTHROPIC_API_KEY`. APIs change often, so check the [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python) before shipping.

After reading

## What you'll learn and where it leads

The code is specific to one support agent. The ideas behind it apply to every agent you'll build or evaluate.

### Nine takeaways

**The model is one block of eight.**Most of an agent's quality comes from the harness: context, tools and limits.

**An agent is a loop.**Read, request a tool or answer, repeat. Understand `stop_reason` and the rest follows.

**Context is a design job.**Give only what this request needs, label each source, and keep per-user data out of the system prompt.

**Tools are an interface for a model.**Plain names, descriptions that say when to use them, and errors that say what to do next.

**Prompts guide; code enforces.**Money limits, permissions and identity belong in hooks and tool code, never only in the prompt.

**Memory has layers.**Context window, session and long-term facts each need their own home and their own rules.

**Measure before you improve.**Log tool calls, record cost and turns, and rerun evals after every change.

**Design the failure path.**Structured output, a success check, and a human handoff that carries the session.

**Start simple.**A fixed workflow is often enough. Add autonomy only when the steps can't be predicted.

### What a reader comes away with

**A mental model**

You can spot the eight blocks in any agent product, so new frameworks become variations rather than new subjects.

**Practical skills**

Write the loop by hand, build MCP tools, write a scoped system prompt, enforce policy with hooks, and run evals.

**Engineering judgment**

You know the failure modes that separate a demo from something customers can safely use.

**A reusable skeleton**

Swap the tools and policy, and the refund agent becomes a helpdesk, sales or operations agent.

### Where this path can lead

**Deeper agent engineering**

- Multi-agent systems: an orchestrator handing work to specialist subagents
- Long-running agents with checkpoints that survive restarts
- Domain coding agents with sandboxes and test loops

**Reliability and evaluation**

- Eval engineering: test sets, model-graded checks, regression runs
- Tracing, cost tuning and production monitoring
- Prompt-injection defense, permission design and audit trails

**Integration and platforms**

- MCP servers that connect company systems to any agent
- Internal agent platforms with shared guardrails and logging

**Roles**

- Hands-on: building and shipping agents, or building them for clients
- Product: knowing what's feasible, what's risky and how to measure it
- Founding: most vertical AI products are a domain harness around a model

AI / agent engineerApplied AI engineerAI product managerFounder

### A six-step practice path

1. **Build the refund agent with mock data.**Then break each block on purpose and watch what changes.
2. **Point it at a real problem.**Something from your own work, with your own tools and rules.
3. **Add a triage agent.**Route requests to specialists to learn orchestration.
4. **Write an MCP server.**Wrap a system you already use and connect it.
5. **Write 20 eval cases.**Improve the agent until they all pass, then keep them passing.
6. **Ship it to a few real users.**Read the logs. That's where the real lessons are.

Agent Harness Field Guide · Code targets the Claude Agent SDK for Python as documented in October 2026.