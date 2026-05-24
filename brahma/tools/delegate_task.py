"""Bootstrap tool: delegate_task — spawn a child Brahma agent."""

from __future__ import annotations

from brahma.tools.registry import ToolRegistry, registry

DELEGATE_TASK_SCHEMA = {
    "description": (
        "Spawn a child Brahma agent to work on a subtask independently."
        " The child gets bootstrap tools and runs with its own context."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "What the child agent should accomplish.",
            },
            "context": {
                "type": "string",
                "description": "Background information for the child agent.",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Tool names to give the child agent."
                    " Defaults to all bootstrap tools if omitted."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Provider:model_name for the child."
                    " Defaults to deepseek:deepseek-chat if omitted."
                ),
            },
        },
        "required": ["goal"],
    },
}


def delegate_task(
    goal: str,
    context: str = "",
    tools: list[str] | None = None,
    model: str = "",
) -> str:
    """Spawn a child agent to work on a task independently.

    This is the spawn meta-capability. In MVP, it creates a new Agent
    instance with a subset of tools and runs it in-process.
    Future: subprocess/K8s Job.
    """
    # Avoid circular import
    from brahma.agent import Agent
    from brahma.bootstrap import BOOTSTRAP_PROMPT
    from brahma.tools import bootstrap_tools

    full_tools = bootstrap_tools()
    child_tools = ToolRegistry()
    requested = tools if tools is not None else full_tools.list_tools()

    for tool_name in requested:
        entry = full_tools.get_entry(tool_name)
        if entry is not None:
            child_tools.register(
                name=tool_name,
                schema=entry.schema,
                handler=entry.handler,
                check_fn=entry.check_fn,
            )

    child = Agent(
        system_prompt=BOOTSTRAP_PROMPT,
        model=model or "deepseek-v4-pro",
        tools=child_tools,
        max_turns=30,
    )

    task_prompt = f"GOAL: {goal}"
    if context:
        task_prompt += f"\n\nCONTEXT:\n{context}"

    return child.run(task_prompt)


registry.register(
    name="delegate_task",
    schema=DELEGATE_TASK_SCHEMA,
    handler=delegate_task,
)
