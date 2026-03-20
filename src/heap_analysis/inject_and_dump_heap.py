import importlib.resources as res
import tempfile
from subprocess import Popen, PIPE

DEFAULT_DUMP_FILE = "/tmp/dump.jsonl"


def dump_heap_from_pid(pid, output_file=DEFAULT_DUMP_FILE):
    """Dump the heap of a running Python process given its PID."""
    with res.open_text("heap_analysis", "dump_heap.py") as f:
        code = f.read()

    code = code.replace(DEFAULT_DUMP_FILE, output_file)
    script_file = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
    script_file.write(code)
    script_file.close()
    try:
        from sys import remote_exec

        remote_exec(pid, script_file.name)
    except ImportError:
        print("remote_exec not available, falling back to gdb method")
        gdb_cmds = [
            "(char *) PyGILState_Ensure()",
            '(void) PyRun_SimpleString("'
            rf'exec(open(\"{script_file.name}\").read())")',
            "(void) PyGILState_Release($1)",
        ]
        p = Popen(
            [
                "gdb",
                "-p",
                str(pid),
                "--batch",
                *(f"--eval-command=call {cmd}" for cmd in gdb_cmds),
            ],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )
        out, err = p.communicate()
        print(out)
        print(err)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dump the heap of a running Python process."
    )
    parser.add_argument("pid", type=int, help="PID of the target Python process")
    parser.add_argument(
        "--output-file",
        "-o",
        default=DEFAULT_DUMP_FILE,
        help=f"Path to output file (default: {DEFAULT_DUMP_FILE})",
    )
    args = parser.parse_args()
    dump_heap_from_pid(args.pid, args.output_file)
