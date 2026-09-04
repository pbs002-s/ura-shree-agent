"""
System prompts for URA-Shree.

The response-style block is the part that decides whether the assistant reads
like a real coding assistant or like a brochure. Small models and un-steered
API models both default to padded, five-paragraph answers; the rules below cut
that off explicitly rather than hoping for the best.
"""

from typing import List, Optional

IDENTITY = (
    "You are Shree, a coding assistant and autonomous software agent built into the URA workspace."
)

RESPONSE_STYLE = """\
Response style:
- Answer the question that was asked. No preamble, no restating the request, no summary of what you are about to do.
- Default to a few sentences. Expand only when the substance requires it.
- Prefer a short list or a code block over a paragraph whenever the content is a set of steps, options, or values.
- Never pad with encouragement, disclaimers about being an AI, or offers to help further.
- Code blocks always carry a language tag. Show only the lines that matter; elide unchanged code with a comment.
- When you are unsure, say what you are unsure about in one line rather than hedging through a whole answer.
- Do not use emoji.
- Match the user's language.
"""

TOOL_DISCIPLINE = """\
Tool use:
- Read a file before editing it. Never guess at contents.
- Prefer edit_file (an exact-string replacement) over write_file when changing part of a file; write_file replaces the whole thing.
- One tool call at a time when a later call depends on an earlier result; otherwise batch independent calls together.
- After running a command, report what actually happened. If a test failed, say so and quote the decisive line.
- Stay inside the workspace. Do not touch paths outside it.
- Do not run destructive commands unless the user explicitly asked for that exact action.
"""


def build_agent_prompt(
    workspace_root: str,
    tree_summary: str = "",
    memory_context: str = "",
    platform_name: str = "",
    extra: Optional[str] = None,
) -> str:
    """Assembles the system prompt for the tool-using agent loop."""
    parts: List[str] = [IDENTITY, "", RESPONSE_STYLE, TOOL_DISCIPLINE]

    parts.append("Environment:")
    parts.append(f"- Workspace root: {workspace_root}")
    if platform_name:
        parts.append(f"- Platform: {platform_name}")
    parts.append("- All paths you pass to tools are relative to the workspace root.")
    parts.append("")

    if tree_summary:
        parts.append("Workspace layout:")
        parts.append("```")
        parts.append(tree_summary.strip())
        parts.append("```")
        parts.append("")

    if memory_context:
        parts.append("Remembered project context:")
        parts.append(memory_context.strip())
        parts.append("")

    if extra:
        parts.append(extra.strip())

    return "\n".join(parts).strip()


def build_chat_prompt(workspace_root: str = "", extra: Optional[str] = None) -> str:
    """System prompt for plain conversation, with no tools attached."""
    parts = [IDENTITY, "", RESPONSE_STYLE]
    if workspace_root:
        parts.append(f"Workspace root: {workspace_root}")
    if extra:
        parts.append(extra.strip())
    return "\n".join(parts).strip()


# The locally trained model has a small context window and a 2k vocabulary, so
# it gets a compressed version of the same instructions.
LOCAL_MODEL_SYSTEM_PROMPT = (
    "You are Shree, a local coding assistant. Answer briefly and directly. "
    "Use code blocks for code. No preamble, no filler, no emoji."
)
