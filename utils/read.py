import argparse
import os
import sys
import requests
import mimetypes
import tiktoken
import time
import pydantic

startingTime = time.perf_counter()

MAX_CHUNK_SIZE = 20 * 1024
IGNORED_FOLDERS = [".git", "__pycache__", ".idea", ".vscode", ".DS_Store", ".pytest_cache", "node_modules", "venv", "env"]
IGNORED_FILES = ["README.md", "LICENSE", ".gitignore", ".env", ".jar", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".webp", ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".mp3", ".wav", ".ogg", ".flac", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]

# TODO: implement folder level reading to reduce token count on large folders

def getTokenCount(text): # NOTE: estimation, not exact. might switch to huggingface tokenizer if response time allows
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    return len(tokens)

def isText(path):
    # Check ignored file extensions
    for ext in IGNORED_FILES:
        if path.endswith(ext):
            return False

    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("text"):
        return True
    try:
        with open(path, 'r', encoding='utf-8') as file:
            file.read(1024)
        return True
    except Exception:
        return False

def getFiles(paths):
    files = []
    for path in paths:
        if os.path.isfile(path) and isText(path):
            files.append(path)
        elif os.path.isdir(path):
            for root, dirnames, filenames in os.walk(path):
                dirnames[:] = [dirname for dirname in dirnames if dirname not in IGNORED_FOLDERS]
                for filename in filenames:
                    fullPath = os.path.join(root, filename)
                    if isText(fullPath):
                        files.append(fullPath)
    return sorted(files)

def chunk(path):
    with open(path, encoding='utf-8', errors='ignore') as file:
        data = file.read()
    if len(data) <= MAX_CHUNK_SIZE:
        return [data]
    return [data[i:i+MAX_CHUNK_SIZE] for i in range(0, len(data), MAX_CHUNK_SIZE)]

def buildMessage(instruction, prompt, chunks):
    messages = [
        {"role": "system", "content": instruction.strip()},
        {"role": "system", "content": "Only answer based on the provided file contents; do not guess or bring in outside info."}
    ]
    for filepath, chunks in chunks.items():
        for idx, chunk in enumerate(chunks):
            label = f"{filepath}"
            if len(chunks) > 1:
                label += f" [chunk {idx+1}/{len(chunks)}]"
            messages.append({
                "role": "user",
                "content": f"File: {label}\n```\n{chunk}\n```"
            })
    messages.append({"role": "user", "content": "User prompt: " + prompt.strip()})
    return messages

def startTimer():
    global startingTime
    startingTime = time.perf_counter()

def getElapsedTime():
    global startingTime
    elapsedTime = time.perf_counter() - startingTime
    return elapsedTime

def main():
    parser = argparse.ArgumentParser(description="Send files + prompt to Ollama LLM with chunking")
    parser.add_argument("model", help="Ollama Model name")
    parser.add_argument("instruction", help="System instruction for the model")
    parser.add_argument("prompt", help="User prompt/question")
    parser.add_argument("paths", nargs="+", help="File(s) or directory(ies) to include")
    args = parser.parse_args()

    chunks = {}
    for filepath in getFiles(args.paths):
        chunks[filepath] = chunk(filepath)

    """ print("Files:")
    for filepath in chunks:
        print(" -", filepath) """

    message = buildMessage(args.instruction, args.prompt, chunks)
    request_payload = {
        "model": args.model, 
        "messages": message,
        "temperature": 0.0
    }

    print("Token count:", getTokenCount(str(request_payload)))

    startTimer()
    response = requests.post("https://ollama.themajorones.dev/v1/chat/completions", json=request_payload)   # TODO: use env or arg
    response.raise_for_status()
    response_json = response.json()
    answer = response_json["choices"][0]["message"]["content"]
    elapsedTime = getElapsedTime()

    print("\n>> Response took :\n", elapsedTime, "seconds \n", answer)

if __name__ == "__main__":
    main()