"""A simple Flask GUI, with templated HTML with very basic styling, that lets the user explore a heap dump created by dump_heap.py."""

from werkzeug.exceptions import NotFound
from ntpath import basename
import os

from flask import Flask, request, redirect, url_for, render_template
from .heap_dump_explorer import HeapDumpExplorer, ObjectSummary

DUMPS_DIR = os.getenv("DUMPS_DIR", "/tmp/dumps")


def create_app():
    os.makedirs(DUMPS_DIR, exist_ok=True)
    loaded_dumps: dict[str, HeapDumpExplorer] = {
        basename(path)[:-5]: HeapDumpExplorer(f"{DUMPS_DIR}/{path}")
        for path in os.listdir(DUMPS_DIR)
        if path.endswith(".lmdb")
    }

    def get_dump(dump_name) -> HeapDumpExplorer:
        explorer = loaded_dumps.get(dump_name)
        if not explorer:
            raise NotFound(f"Dump '{dump_name}' not found")
        return explorer

    app = Flask(__name__)

    @app.route("/")
    def index():
        dump_names = list(loaded_dumps.keys())
        return render_template("index.html", dump_names=dump_names)

    @app.route("/upload_dump", methods=["POST"])
    def upload_dump():
        dump_name = request.form["dump_name"]
        dump_file = request.files["dump_file"]
        explorer = HeapDumpExplorer(f"{DUMPS_DIR}/{dump_name}.lmdb")
        explorer.import_lines(dump_file)
        loaded_dumps[dump_name] = explorer
        return redirect(url_for("explore_dump", dump_name=dump_name))

    @app.route("/explore/<dump_name>")
    def explore_dump(dump_name):
        explorer = get_dump(dump_name)
        # For simplicity, just show a count of objects by type
        type_counts: list[tuple[str, int]] = explorer.get_type_counts()
        return render_template(
            "explore.html", dump_name=dump_name, type_counts=type_counts
        )

    @app.route("/explore/<dump_name>/type/<type_name>")
    def explore_type(dump_name, type_name):
        explorer = get_dump(dump_name)
        objects: list[ObjectSummary] = explorer.get_objects_by_type(type_name)
        if request.args.get('sort') == 'size':
            objects.sort(key=lambda o: o.size, reverse=True)
        elif request.args.get('sort') == 'subtree_size':
            objects.sort(key=lambda o: o.subtree_size, reverse=True)
        return render_template(
            "type.html", dump_name=dump_name, type_name=type_name, objects=objects
        )

    @app.route("/explore/<dump_name>/object/<int:obj_id>")
    def explore_object(dump_name, obj_id):
        explorer = get_dump(dump_name)
        obj = explorer.get_object(obj_id, references="summaries")
        if not obj:
            raise NotFound(f"Object with ID {obj_id} not found in dump '{dump_name}'")
        return render_template("object.html", dump_name=dump_name, obj=obj)

    return app
