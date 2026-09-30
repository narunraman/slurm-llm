"""`sllm query`: one-off questions and an interactive chat through the proxy."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape

console = Console()


class ThinkingTagProcessor:
    """Dims <think>...</think> text in streamed output (reasoning models)."""

    def __init__(self, is_reasoning_model: bool = False):
        # reasoning models may start thinking without an opening tag
        self.in_thinking = is_reasoning_model
        self.buffer = ""

    def process_chunk(self, chunk: str) -> str:
        """Rich markup for the text that is safe to print (a split tag stays buffered)."""
        self.buffer += chunk
        output = ""
        while True:
            tag = "</think>" if self.in_thinking else "<think>"
            idx = self.buffer.find(tag)
            if idx == -1:
                keep = len(tag) - 1  # the start of a tag may still be arriving
                text, self.buffer = self.buffer[:-keep] if len(self.buffer) > keep else "", self.buffer[-keep:]
                output += self._format(text)
                return output
            output += self._format(self.buffer[:idx]) + f"[dim]{escape(tag)}[/dim]"
            self.buffer = self.buffer[idx + len(tag):]
            self.in_thinking = not self.in_thinking

    def flush(self) -> str:
        text, self.buffer = self.buffer, ""
        return self._format(text)

    def _format(self, text: str) -> str:
        if not text:
            return ""
        return f"[dim]{escape(text)}[/dim]" if self.in_thinking else escape(text)


def _is_reasoning_model(model_name: str) -> bool:
    try:
        from .config import ConfigLoader

        model = ConfigLoader().resolve_model(model_name.rsplit(":", 1)[0])
        return bool(model and "reasoning" in model.tags)
    except Exception:
        return False


def _complete(client, model: str, messages: list, settings: dict, stream: bool) -> str:
    """Send one chat request and print the answer. Returns the answer text."""
    processor = ThinkingTagProcessor(_is_reasoning_model(model))
    console.print(f"[bold]{escape(model)}:[/bold] ", end="")
    if stream:
        answer = ""
        for chunk in client.chat.completions.create(model=model, messages=messages, stream=True, **settings):
            if chunk.choices and chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
                console.print(processor.process_chunk(chunk.choices[0].delta.content), end="")
    else:
        response = client.chat.completions.create(model=model, messages=messages, **settings)
        answer = (response.choices[0].message.content or "") if response.choices else ""
        console.print(processor.process_chunk(answer), end="")
    console.print(processor.flush())
    return answer


def query_once(client, model: str, prompt: str, system: str | None, settings: dict, stream: bool) -> None:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    _complete(client, model, messages, settings, stream)


HELP = """Commands:
  /system TEXT      set the system prompt
  /clear            forget the conversation
  /model NAME       switch model
  /set KEY VALUE    temperature, max_tokens or top_p
  /show             show the settings
  /exit             quit"""


def chat(client, model: str, system: str | None, settings: dict, stream: bool, proxy_url: str) -> None:
    """Interactive chat; the conversation is kept until /clear."""
    import readline  # noqa: F401  (line editing and history for input())

    messages = [{"role": "system", "content": system}] if system else []
    console.print(f"[bold]Chatting with {escape(model)}[/bold] [dim](/help for commands, /exit to quit)[/dim]")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            console.print()
            return
        except KeyboardInterrupt:
            console.print("\n[dim]/exit to quit[/dim]")
            continue
        if not text:
            continue
        if text.startswith("/"):
            command, _, arg = text[1:].partition(" ")
            arg = arg.strip()
            if command in ("exit", "quit"):
                return
            if command == "help":
                console.print(HELP)
            elif command == "system":
                messages = [m for m in messages if m["role"] != "system"]
                if arg:
                    messages.insert(0, {"role": "system", "content": arg})
                console.print("[green]✓ system prompt set[/green]" if arg else "[green]✓ system prompt cleared[/green]")
            elif command == "clear":
                messages = [m for m in messages if m["role"] == "system"]
                console.print("[green]✓ conversation cleared[/green]")
            elif command == "model" and arg:
                model = arg
                console.print(f"[green]✓ now chatting with {escape(model)}[/green]")
            elif command == "set" and len(arg.split()) == 2:
                key, value = arg.split()
                key = {"temp": "temperature"}.get(key, key)
                if key not in ("temperature", "max_tokens", "top_p"):
                    console.print("[red]settings: temperature, max_tokens, top_p[/red]")
                    continue
                try:
                    settings[key] = int(value) if key == "max_tokens" else float(value)
                except ValueError:
                    console.print(f"[red]invalid value for {key}[/red]")
            elif command == "show":
                console.print(f"model {escape(model)}, {settings}, {len(messages)} message(s)")
            else:
                console.print(f"[red]unknown command /{escape(command)}[/red] (/help)")
            continue

        messages.append({"role": "user", "content": text})
        try:
            messages.append({"role": "assistant", "content": _complete(client, model, messages, settings, stream)})
        except KeyboardInterrupt:
            messages.pop()
            console.print("\n[dim]interrupted[/dim]")
        except Exception as e:
            messages.pop()
            console.print(f"\n[red]Error: {escape(str(e))}[/red]")
