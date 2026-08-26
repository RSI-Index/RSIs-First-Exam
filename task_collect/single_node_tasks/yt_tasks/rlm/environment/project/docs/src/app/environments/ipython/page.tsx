import { CodeBlock } from "@/components/CodeBlock";
import { Table } from "@/components/Table";

export default function IPythonPage() {
  return (
    <div>
      <h1 className="text-3xl font-bold mb-4">IPythonREPL</h1>

      <p className="text-xl text-muted-foreground mb-6 leading-relaxed">
        <strong className="text-foreground">IPythonREPL</strong> executes code inside a real{" "}
        <a href="https://ipython.org/" className="text-primary hover:underline font-medium">IPython</a>{" "}
        session instead of plain <code className="px-1.5 py-0.5 rounded bg-muted text-foreground text-sm font-semibold">exec()</code>.
        It supports two kernel modes — an <strong className="text-foreground">in-process</strong> shell that
        runs in the same Python process as the RLM (the default, fastest), and a{" "}
        <strong className="text-foreground">subprocess</strong> kernel that runs a real{" "}
        <code className="px-1.5 py-0.5 rounded bg-muted text-foreground text-sm font-semibold">ipykernel</code> in a separate
        Python process for hard cell timeouts and full namespace isolation from the RLM host.
        Both modes give the LM access to IPython&apos;s full surface (cell magics, rich
        repr, line tracebacks).
      </p>

      <p className="text-muted-foreground mb-4">
        <strong>Prerequisite:</strong> install the optional extra:
      </p>
      <CodeBlock language="bash" code={`pip install 'rlms[ipython]'
# or with uv:
# uv pip install -e ".[ipython]"`} />

      <CodeBlock code={`from rlm import RLM

# In-process (default kernel_mode): same process, fast.
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini"},
    environment="ipython",
    environment_kwargs={
        "kernel_mode": "in_process",
        "cell_timeout": 30,         # SIGALRM-based; Unix main thread only
    },
)

# Subprocess: separate Python process, hard timeouts, full isolation.
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-mini"},
    environment="ipython",
    environment_kwargs={
        "kernel_mode": "subprocess",
        "cell_timeout": 30,         # Hard guarantee via interrupt_kernel
        "startup_timeout": 60,
        "max_concurrent_subcalls": 4,
    },
)`} />

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">Arguments</h2>
      <Table
        headers={["Argument", "Type", "Default", "Description"]}
        rows={[
          [<code key="1">kernel_mode</code>, <code key="2">&quot;in_process&quot; | &quot;subprocess&quot;</code>, <code key="3">&quot;in_process&quot;</code>, "Where the IPython session runs"],
          [<code key="4">cell_timeout</code>, <code key="5">float | None</code>, <code key="6">None</code>, "Per-cell timeout in seconds; None disables"],
          [<code key="7">startup_timeout</code>, <code key="8">float</code>, <code key="9">60.0</code>, "Subprocess kernel boot timeout"],
          [<code key="10">subcall_timeout</code>, <code key="11">float | None</code>, <code key="12">None</code>, "Per-request kernel→broker socket timeout (subprocess)"],
          [<code key="13">max_concurrent_subcalls</code>, <code key="14">int</code>, <code key="15">4</code>, "Global cap on concurrent subcall_fn invocations"],
          [<code key="16">setup_code</code>, <code key="17">str</code>, <code key="18">None</code>, "Code to run at initialization"],
          [<code key="19">custom_tools</code>, <code key="20">dict</code>, <code key="21">None</code>, "Functions / values injected into the namespace"],
        ]}
      />

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">In-process vs. subprocess</h2>
      <Table
        headers={["", "in_process", "subprocess"]}
        rows={[
          ["Process", "Same as host", "Separate Python via ipykernel"],
          ["Subcall path", "Direct Python call", "TCP broker (4-byte length-prefixed JSON)"],
          [<><code key="t1">cell_timeout</code></>, "Best-effort SIGALRM (Unix, main thread)", <>Hard, via <code key="t2">interrupt_kernel</code></>],
          ["Cell magics (%%timeit, …)", "Yes", "Yes"],
          [<><code key="t3">input()</code></>, "Disabled (raises)", <>Disabled (<code key="t4">allow_stdin=False</code>)</>],
          ["Isolation from host", "Shares stdout/stderr/cwd/SIGALRM", "Full process isolation"],
          ["Custom tool injection", "Direct namespace inject", <>Pickled with <code key="t5">dill</code> over ZMQ</>],
        ]}
      />

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
      <h3 className="text-lg font-medium mt-4 mb-2">In-process</h3>
      <ol className="list-decimal list-inside text-muted-foreground space-y-1 mb-4">
        <li>Creates a fresh <code>InteractiveShell</code> with a per-instance user module (so multiple in-process REPLs don&apos;t share <code>sys.modules[&apos;__main__&apos;]</code>) and IPython&apos;s history database disabled.</li>
        <li>Injects scaffold helpers (<code>llm_query</code>, <code>rlm_query</code>, <code>FINAL_VAR</code>, <code>SHOW_VARS</code>) and a stubbed <code>input()</code> into <code>user_ns</code>.</li>
        <li><code>execute_code</code> runs each cell via <code>shell.run_cell</code> under an <code>RLock</code>; <code>cell_timeout</code> is enforced with <code>SIGALRM</code> + <code>setitimer</code>.</li>
        <li><code>rlm_query</code> calls <code>subcall_fn</code> directly, gated by a per-instance semaphore so kernel-side fan-out can&apos;t exceed <code>max_concurrent_subcalls</code>.</li>
      </ol>

      <h3 className="text-lg font-medium mt-4 mb-2">Subprocess</h3>
      <ol className="list-decimal list-inside text-muted-foreground space-y-1 mb-4">
        <li>Starts a TCP broker on <code>127.0.0.1:0</code> (ephemeral port).</li>
        <li>Launches an <code>ipykernel</code> subprocess pinned to the host&apos;s <code>sys.executable</code> (so it inherits the same site-packages — important for <code>dill</code>, custom imports, etc.).</li>
        <li>Bootstraps kernel-side scaffold helpers that route <code>llm_query</code> to the LM Handler and <code>rlm_query</code> / <code>FINAL_VAR</code> to the broker over the 4-byte-prefixed JSON protocol.</li>
        <li>Each user cell first sets a unique <code>_RLM_CURRENT_CELL</code> cell-id in the kernel via a separate <code>execute_interactive</code> call (so cell magics still work). Every broker request carries this id.</li>
        <li><code>cell_timeout</code> is enforced by <code>kc.execute_interactive(timeout=…)</code> + <code>km.interrupt_kernel()</code>. Subcall completions whose <code>subcall_fn</code> finishes <em>after</em> the originating cell timed out are stored under that cell&apos;s id and discarded as stale on the next drain — they aren&apos;t misattributed to a later cell.</li>
      </ol>

      <pre className="text-sm">{`┌──────────────────────────────────────────┐
│ Host (RLM process)                       │
│  ┌─────────────┐ Socket ┌──────────────┐ │
│  │ Subcall     │◄──────►│  LM Handler  │ │
│  │ broker      │        └──────────────┘ │
│  │ (TCP + JSON)│                         │
│  └─────┬───────┘                         │
└────────┼─────────────────────────────────┘
         │ ZMQ (jupyter_client)
┌────────┼─────────────────────────────────┐
│ ipykernel subprocess                     │
│  ┌─────▼────────┐                        │
│  │  IPython     │ rlm_query() / FINAL_VAR│
│  │  user_ns     │ → broker over TCP      │
│  │  (cell_id    │ llm_query() → LM Handler│
│  │   tagged)    │                        │
│  └──────────────┘                        │
└──────────────────────────────────────────┘`}</pre>

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">Notable behavior</h2>
      <ul className="list-disc list-inside text-muted-foreground space-y-2">
        <li><strong className="text-foreground">Per-instance serialization.</strong> <code>execute_code</code> takes an <code>RLock</code>, so concurrent calls from different threads are serialized within an instance.</li>
        <li><strong className="text-foreground">Global subcall cap.</strong> <code>max_concurrent_subcalls</code> bounds <em>total</em> in-flight <code>subcall_fn</code> invocations on the instance — even if user code spawns kernel-side threads that each fan out a batch.</li>
        <li><strong className="text-foreground">Reentry guard.</strong> If <code>subcall_fn</code> calls <code>execute_code</code> back on the parent REPL (or a cell traverses <code>rlm_query.__self__.execute_code(…)</code> in in-process mode), the call raises <code>RuntimeError</code> instead of deadlocking the cell lock or corrupting the in-flight cell&apos;s tracking. <code>subcall_fn</code> should spawn a child REPL.</li>
        <li><strong className="text-foreground">Cell-id attribution.</strong> Subcall completions are tagged with the originating cell&apos;s id so a slow <code>subcall_fn</code> that finishes after its cell timed out is never counted under a later cell. Long-lived kernel threads that call <code>rlm_query</code> after their spawning cell ends will, however, be tagged with whatever cell is active at call time.</li>
        <li><strong className="text-foreground">In-process is not isolated.</strong> Two in-process instances each get a unique <code>__main__</code> substitute, but they still share the host&apos;s stdout/stderr/cwd/SIGALRM. Use <code>subprocess</code> if you need true isolation.</li>
      </ul>

      <hr className="my-8 border-border" />

      <h2 className="text-2xl font-semibold mb-4">When to use which mode</h2>
      <ul className="list-disc list-inside text-muted-foreground space-y-1.5">
        <li><strong className="text-foreground">in_process</strong> — fastest path, no IPC, fine for trusted code, development, short-lived cells. <code>cell_timeout</code> is best-effort (Unix main thread only).</li>
        <li><strong className="text-foreground">subprocess</strong> — when you need a hard <code>cell_timeout</code> guarantee, or want full namespace / signal / cwd isolation between the LM&apos;s code and the RLM host.</li>
      </ul>
    </div>
  );
}
