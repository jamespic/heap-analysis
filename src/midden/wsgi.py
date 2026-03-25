"""A simple Flask GUI, with templated HTML with very basic styling, that lets the user explore a heap dump created by dump_heap.py."""

from werkzeug.exceptions import NotFound
from ntpath import basename
import os

from flask import Flask, request, redirect, url_for, render_template
from .heap_dump_explorer import HeapDumpExplorer

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
        if dump_name in loaded_dumps:
            return f"Dump with name '{dump_name}' already exists", 409
        if "/" in dump_name or "\\" in dump_name:
            return "Invalid dump name", 400
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
        page = request.args.get("page", 1, type=int)
        page_zero_indexed = max(page - 1, 0)
        explorer = get_dump(dump_name)
        sort = request.args.get("sort", "id")
        match sort:
            case "size":
                objects = explorer.get_objects_by_type_ordered_by_size(
                    type_name, page=page_zero_indexed, subtree_size=False
                )
            case "subtree_size":
                objects = explorer.get_objects_by_type_ordered_by_size(
                    type_name, page=page_zero_indexed, subtree_size=True
                )
            case "id" | _:
                objects = explorer.get_objects_by_type(
                    type_name, page=page_zero_indexed
                )
        total_pages = explorer.get_page_count_for_type(type_name)
        return render_template(
            "type.html",
            dump_name=dump_name,
            type_name=type_name,
            objects=objects,
            page=page,
            total_pages=total_pages,
            sort=sort,
        )

    @app.route("/explore/<dump_name>/object/<int:obj_id>")
    def explore_object(dump_name, obj_id):
        explorer = get_dump(dump_name)
        obj = explorer.get_object(obj_id)
        if not obj:
            raise NotFound(f"Object with ID {obj_id} not found in dump '{dump_name}'")
        return render_template("object.html", dump_name=dump_name, obj=obj)
    
    @app.route("/explore/<dump_name>/find_path")
    def find_path(dump_name):
        explorer = get_dump(dump_name)
        from_id = request.args.get("from_id", type=int)
        to_id = request.args.get("to_id", type=int)
        if from_id is None or to_id is None:
            return "Missing from_id or to_id query parameters", 400
        path = explorer.find_path_between_objects(from_id, to_id)
        if path is None:
            return f"No path found from object {from_id} to object {to_id}", 404
        return render_template(
            "path.html",
            dump_name=dump_name,
            from_id=from_id,
            to_id=to_id,
            path=path,
        )

    return app

def main():
    app = create_app()
    app.run()

if __name__ == "__main__":
    main()