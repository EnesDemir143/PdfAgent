from langchain_core.messages import AIMessage, ToolMessage


async def execute_tools(tool_call: AIMessage, tools) -> ToolMessage:
    tool_id = tool_call[0]['name']
    tool_args = tool_call[0]['args']
    tool_out = await tools[tool_id](**tool_args)
    return ToolMessage(
        content=f"{tool_out}",
        tool_call_id=tool_call.tool_calls[0]["id"]
    )