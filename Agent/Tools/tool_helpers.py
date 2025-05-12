from langchain_core.messages import AIMessage, ToolMessage


# execute_tools fonksiyonu
async def execute_tools(tool_call: AIMessage, tools) -> ToolMessage:
    tool_name = tool_call.tool_calls[0]["name"]
    tool_args = tool_call.tool_calls[0]["args"]
    tool_out = await tools[tool_name](**tool_args)
    return ToolMessage(
        content=f"{tool_out}",
        tool_call_id=tool_call.tool_calls[0]["id"]
    )