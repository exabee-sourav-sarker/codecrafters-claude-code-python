import argparse
import os
import subprocess
import sys
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
MODEL = "anthropic/claude-haiku-4.5"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read the content of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The file path"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The file path"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    }
                }
            }
        }
    }
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    messages = [
        {
            "role": "user",
            "content": args.p
        }
    ]

  
    while True:
        chat = make_calls(client, messages)
        msg = chat.choices[0].message

        messages.append(msg)
        if not msg.tool_calls:
            break

        for tool in msg.tool_calls:
            arg = json.loads(tool.function.arguments)
            if tool.function.name == "Read":
                content = call_read_func(arg, tool.id)
                messages.append(content)
            

            if tool.function.name == "Write":
                content = call_write_func(arg, tool.id)
                messages.append(content)

            if tool.function.name == "Bash":
                content = call_bash_func(arg, tool.id)
                messages.append(content)
        


    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

def call_read_func(arg, id):
    with open(arg["file_path"], 'r') as f:
        content = f.read()
    return {
        "role": "tool",
        "tool_call_id": id,
        "content": content
    }

def call_bash_func(arg, id):
    content = subprocess.run(arg["command"].split(), capture_output=True, text=True)

    return {
        "role": "tool",
        "tool_call_id": id,
        "name": "Bash",
        "content": f"{content.stdout} {content.stderr}"
    }


def call_write_func(arg, id):
    with open(arg["file_path"], 'w') as f:
        f.write(arg["content"])

    return {
        "role": "tool",
        "tool_call_id": id,
        "content": f"Content written to {arg['file_path']}"
    }

def make_calls(client: OpenAI, messages: list):

    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS
    )


if __name__ == "__main__":
    main()
