"""Bridge the pinned agent engine's history format to GGUF chat templates."""
import json

from agentic_loop.providers.openai_compatible import OpenAICompatibleChatModel
from agentic_loop.server import AgentServerApp


class LlamaChatModel(OpenAICompatibleChatModel):
    def __init__(self, model, base_url, timeout_s=300):
        super().__init__(model, base_url, api_key="", timeout_s=timeout_s)
        self._calls = {}

    def respond(self, messages, tools, state_summary):
        response = super().respond(messages, tools, state_summary)
        for call in response.tool_calls:
            self._calls[call.id] = call
        return response

    def _messages(self, messages):
        # AgenticLocal adds policy/state system messages after the user message.
        # Qwen and other Jinja templates require one leading system message.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        rendered = [{"role": "system", "content": system}] if system else []
        for message in messages:
            if message.role == "system":
                continue
            if message.role != "tool":
                rendered.append({"role": message.role, "content": message.content})
                continue
            call = self._calls.get(message.tool_call_id)
            if call:
                # The pinned engine records a textual request + tool result;
                # restore the original native tool call for the next model turn.
                if rendered and rendered[-1].get("content") == f"Tool call requested: {call.name}":
                    rendered.pop()
                rendered.append({"role": "assistant", "content": "", "tool_calls": [{
                    "id": call.id, "type": "function", "function": {
                        "name": call.name, "arguments": json.dumps(call.arguments),
                    },
                }]})
                rendered.append({"role": "tool", "tool_call_id": call.id, "content": message.content})
            else:
                # Reopened stored sessions do not retain the original arguments.
                # Preserve the observation without inventing a native tool call.
                rendered.append({"role": "user", "content": f"Previous tool result ({message.name or 'tool'}):\n{message.content}"})
        return rendered

    def _post(self, path, payload):
        return super()._post(path, {**payload, "temperature": 0.2, "max_tokens": 2048,
                                   "chat_template_kwargs": {"enable_thinking": False}})


class LlamaAgentServerApp(AgentServerApp):
    def _make_controller(self, model_selection=None):
        controller = super()._make_controller(model_selection)
        selected = model_selection or self.default_model_selection
        controller.model = LlamaChatModel(selected.model_name, selected.api_base)
        return controller
