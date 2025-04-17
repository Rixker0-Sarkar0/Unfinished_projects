#!/usr/bin/env python3
import argparse
import sqlite3
import subprocess
import psutil
import multiprocessing as mp
import os
import docker
import math
import time
import fitz  # PyMuPDF
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from rich.console import Console
from rich.table import Table

# ─── CONFIGURATION ──────────────────────────────────────────────────────
DEFAULT_MODEL   = "mpt-1b-q4k"
SQLITE_DB       = "rag_omniverse.db"
NODE_MEM_MB     = 200
RAM_CAP_PCT     = 0.9
DOCKER_IMAGE    = "alpine"
AGENT_CYCLES    = 3
# ────────────────────────────────────────────────────────────────────────

console = Console()

# ─── RAG COMPONENT ───────────────────────────────────────────────────────
class RAG:
    def __init__(self, db_path=SQLITE_DB):
        self._init_sqlite(db_path)
        self.embed = HuggingFaceEmbeddings()
        self.index_path = "rag_index.faiss"
        self.index = FAISS.from_texts(["Initial system boot."], self.embed)
        self.index.write_index(self.index_path)
        self.index = FAISS.read_index(self.index_path, self.embed.embed_query, mmap=True)

    def _init_sqlite(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS logs(id INTEGER PRIMARY KEY, text TEXT)")
        self.conn.commit()

    def log(self, text: str):
        self.conn.execute("INSERT INTO logs(text) VALUES(?)", (text,))
        self.conn.commit()
        self.index.add_texts([text])
        self.index.write_index(self.index_path)
        self.index = FAISS.read_index(self.index_path, self.embed.embed_query, mmap=True)

    def retrieve(self, query: str, k: int = 4) -> str:
        results = self.index.similarity_search(query, k=k)
        return "\n".join([r.page_content for r in results])

# ─── FILE HANDLER ────────────────────────────────────────────────────────
def load_reference_text(path: str) -> str:
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif path.endswith(".pdf"):
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    else:
        raise ValueError("Only .txt or .pdf supported.")

# ─── LLM HELPER ──────────────────────────────────────────────────────────
def llm_generate(prompt: str, model: str) -> str:
    try:
        return subprocess.check_output([
            "ollama", "run", model,
            "--quiet", "--command", prompt
        ], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError as e:
        return f"[ERROR] LLM failed: {e}"

# ─── NODE LOGIC ──────────────────────────────────────────────────────────
def node_worker(node_id: int, conn, model: str):
    for cycle in range(AGENT_CYCLES):
        ctx = conn.recv()
        prompt = f"Node {node_id} [Cycle {cycle}]: theorize next system insight or construct."
        response = llm_generate(ctx + "\n\n" + prompt, model)
        conn.send(response)

def spin_docker(node_id: int):
    client = docker.from_env()
    client.containers.run(
        DOCKER_IMAGE,
        command="sleep 10",
        detach=True,
        mem_limit=f"{NODE_MEM_MB}m",
        name=f"node-{node_id}",
        auto_remove=True
    )

# ─── UTILITY ─────────────────────────────────────────────────────────────
def get_max_nodes(mem_per_node_mb: int, ram_cap_pct: float = RAM_CAP_PCT) -> int:
    total_ram = psutil.virtual_memory().total // (1024 * 1024)
    safe_ram = math.floor((total_ram * ram_cap_pct) / mem_per_node_mb)
    cores = psutil.cpu_count(logical=False) or 1
    return max(1, min(safe_ram, cores))

def render_table(outputs):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Node", justify="center")
    table.add_column("Cycle", justify="center")
    table.add_column("Output", overflow="fold")

    for node_id, node_logs in outputs.items():
        for i, msg in enumerate(node_logs):
            table.add_row(f"{node_id}", f"{i}", msg[:100] + "..." if len(msg) > 100 else msg)

    console.clear()
    console.print(table)

# ─── MAIN ORCHESTRATOR ───────────────────────────────────────────────────
def orchestrate(model: str, dockerize: bool, mem_per_node: int, max_nodes: int):
    rag = RAG()
    count = min(get_max_nodes(mem_per_node), max_nodes)
    console.log(f"[SYS] Spawning {count} node(s) w/ {AGENT_CYCLES} cycles each.")
    processes, pipes = [], []
    all_outputs = {i: [] for i in range(count)}

    for i in range(count):
        if dockerize:
            spin_docker(i)
        parent_conn, child_conn = mp.Pipe()
        proc = mp.Process(target=node_worker, args=(i, child_conn, model))
        proc.start()
        pipes.append((i, parent_conn))
        processes.append(proc)

    while True:
        user_input = console.input("[bold cyan]/> [/]")
        if user_input.startswith("/ref "):
            filepath = user_input[5:].strip()
            try:
                ref_text = load_reference_text(filepath)
                rag.log(ref_text)
                console.print(f"[green]Loaded reference: {filepath} ({len(ref_text)} chars)[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load: {e}[/red]")
            continue
        elif user_input == "/exit":
            console.print("[yellow]Exiting simulation...[/yellow]")
            break

        for node_id, conn in pipes:
            conn.send(rag.retrieve(user_input or "Continue simulation."))
        for node_id, conn in pipes:
            output = conn.recv()
            rag.log(output)
            all_outputs[node_id].append(output)
        render_table(all_outputs)

    for p in processes:
        p.join()

# ─── ENTRY ───────────────────────────────────────────────────────────────
def main():
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="In‑Mem Omniverse Simulator CLI")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument("--docker", action="store_true", help="Enable Docker per node")
    parser.add_argument("--max-nodes", type=int, default=8, help="Max number of nodes")
    parser.add_argument("--node-mem", type=int, default=NODE_MEM_MB, help="Memory per node (MB)")
    args = parser.parse_args()

    orchestrate(args.model, args.docker, args.node_mem, args.max_nodes)

if __name__ == "__main__":
    main()
