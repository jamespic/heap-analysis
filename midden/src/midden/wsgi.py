"""A simple Flask GUI, with templated HTML with very basic styling, that lets the user explore a heap dump created by dump_heap.py."""

import argparse
import os
import pathlib
import shutil
import webbrowser
from ntpath import basename

from cheroot.wsgi import Server as WSGIServer
from flask import Flask, redirect, render_template, request, session, url_for
from midden_analysis import EstimatorPrecision, HeapDumpExplorer, TypeSummary
from werkzeug.exceptions import NotFound

DUMPS_DIR = os.getenv("DUMPS_DIR", "/tmp/dumps")

PRECISION_MAP = {
    "no_estimates": EstimatorPrecision.NoEstimates,
    "low": EstimatorPrecision.Low,
    "medium": EstimatorPrecision.Medium,
    "high": EstimatorPrecision.High,
    "exact": EstimatorPrecision.Exact,
}


def create_app():
    """Create the Flask app and preload any dumps already stored on disk."""
    os.makedirs(DUMPS_DIR, exist_ok=True)
    loaded_dumps: dict[str, HeapDumpExplorer] = {
        basename(path)[:-5]: HeapDumpExplorer(f"{DUMPS_DIR}/{path}")
        for path in os.listdir(DUMPS_DIR)
        if path.endswith(".lmdb")
    }

    def get_dump(dump_name) -> HeapDumpExplorer:
        """Return a loaded dump or raise a 404 for unknown names."""
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
        """Import an uploaded JSONL heap dump into a new LMDB-backed explorer."""
        dump_name = request.form.get("dump_name")
        dump_file = request.files["dump_file"]
        if not dump_name:
            if upload_filename := dump_file.filename:
                dump_name = pathlib.Path(upload_filename).stem
            else:
                dump_name = f"heap_dump_{len(loaded_dumps) + 1}"
        if dump_name in loaded_dumps:
            return f"Heap dump with name '{dump_name}' already exists", 409
        if "/" in dump_name or "\\" in dump_name or dump_name.startswith("."):
            return "Invalid heap dump name", 400
        dump_dir = f"{DUMPS_DIR}/{dump_name}.lmdb"
        precision = request.form.get("estimator_precision", "medium")
        estimator_precision = PRECISION_MAP.get(precision, EstimatorPrecision.Medium)
        os.mkdir(dump_dir)
        try:
            explorer = HeapDumpExplorer(f"{DUMPS_DIR}/{dump_name}.lmdb")
            explorer.import_lines(dump_file, estimator_precision)
        except Exception:
            shutil.rmtree(dump_dir)
            raise
        loaded_dumps[dump_name] = explorer
        return redirect(url_for("explore_dump", dump_name=dump_name))

    @app.route("/explore/<dump_name>")
    def explore_dump(dump_name):
        explorer = get_dump(dump_name)
        # The landing page for a dump is a type summary table.
        type_summaries: list[tuple[str, TypeSummary]] = explorer.get_type_summaries()
        sort_by = request.args.get("sort_by", "count")
        match sort_by:
            case "size":
                type_summaries.sort(key=lambda x: x[1].total_size, reverse=True)
            case "type":
                type_summaries.sort(key=lambda x: x[0])
            case "count" | _:
                type_summaries.sort(key=lambda x: x[1].count, reverse=True)
        return render_template(
            "explore.html", dump_name=dump_name, type_summaries=type_summaries
        )

    @app.route("/explore/<dump_name>/type/<type_name>")
    def explore_type(dump_name, type_name):
        """Show one page of objects for a type, with optional size-based sorting."""
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
        """Show one object together with its references and referrers."""
        explorer = get_dump(dump_name)
        obj = explorer.get_object(obj_id)
        current_from_id = session.get(f"path_finding_from_id:{dump_name}")
        current_to_id = session.get(f"path_finding_to_id:{dump_name}")
        if not obj:
            raise NotFound(f"Object with ID {obj_id} not found in dump '{dump_name}'")
        return render_template(
            "object.html",
            dump_name=dump_name,
            obj=obj,
            current_from_id=current_from_id,
            current_to_id=current_to_id,
        )

    @app.route("/explore/<dump_name>/find_largest_strict_descendant")
    def find_largest_strict_descendant(dump_name):
        """Find the largest strict descendant of an object in the selected dump."""
        explorer = get_dump(dump_name)
        obj_id = request.args.get("obj_id", type=int)
        if obj_id is None:
            return "Missing obj_id query parameter", 400
        largest_descendant_id = explorer.find_largest_strict_descendant(obj_id)
        if largest_descendant_id is None:
            raise NotFound(
                f"No strict descendants found for object ID {obj_id} in dump '{dump_name}'"
            )
        return redirect(
            url_for("explore_object", dump_name=dump_name, obj_id=largest_descendant_id)
        )

    @app.route("/explore/<dump_name>/find_largest_ancestor")
    def find_largest_ancestor(dump_name):
        """Find the largest ancestor of an object in the selected dump."""
        explorer = get_dump(dump_name)
        obj_id = request.args.get("obj_id", type=int)
        if obj_id is None:
            return "Missing obj_id query parameter", 400
        largest_ancestor_id = explorer.find_largest_ancestor(obj_id)
        return redirect(
            url_for(
                "find_path",
                dump_name=dump_name,
                from_id=largest_ancestor_id,
                to_id=obj_id,
            )
        )

    @app.route("/explore/<dump_name>/set_path_finding_endpoint", methods=["POST"])
    def set_path_finding_endpoint(dump_name):
        """Store or complete the pair of object IDs used for path finding."""
        from_id = request.form.get(
            "from_id", session.get(f"path_finding_from_id:{dump_name}"), type=int
        )
        to_id = request.form.get(
            "to_id", session.get(f"path_finding_to_id:{dump_name}"), type=int
        )
        if from_id is not None and to_id is not None:
            del session[f"path_finding_from_id:{dump_name}"]
            del session[f"path_finding_to_id:{dump_name}"]
            return redirect(
                url_for("find_path", dump_name=dump_name, from_id=from_id, to_id=to_id)
            )
        else:
            session[f"path_finding_from_id:{dump_name}"] = from_id
            session[f"path_finding_to_id:{dump_name}"] = to_id

            return redirect(
                url_for("explore_object", dump_name=dump_name, obj_id=from_id or to_id)
            )

    @app.route("/explore/<dump_name>/find_path")
    def find_path(dump_name):
        """Find a reference path between two objects in the selected dump."""
        explorer = get_dump(dump_name)
        from_id = request.args.get("from_id", type=int)
        to_id = request.args.get("to_id", type=int)
        avoid_ids = set(request.args.getlist("avoid_id", type=int))
        if from_id is None or to_id is None:
            return "Missing from_id or to_id query parameters", 400
        path = explorer.find_path_between_objects(
            from_id, to_id, avoiding_ids=avoid_ids
        )
        return render_template(
            "path.html",
            dump_name=dump_name,
            from_id=from_id,
            to_id=to_id,
            path=path,
            avoid_ids=list(avoid_ids),
        )

    @app.route("/explore/<dump_name>/find_largest_common_reachable_object")
    def find_largest_common_reachable_object(dump_name):
        """Find the largest common reachable object between two objects in the selected dump."""
        explorer = get_dump(dump_name)
        obj1_id = request.args.get("obj1_id", type=int)
        obj2_id = request.args.get("obj2_id", type=int)
        if obj1_id is None or obj2_id is None:
            return "Missing obj1_id or obj2_id query parameters", 400
        common_object_id = explorer.find_largest_common_reachable_object(
            obj1_id, obj2_id
        )
        if common_object_id is None:
            return render_template(
                "largest_common_reachable_object.html",
                dump_name=dump_name,
                obj1_id=obj1_id,
                obj2_id=obj2_id,
                common_object=None,
            )
        common_object = explorer.get_object(common_object_id)
        assert common_object is not None, (
            f"Common object ID {common_object_id} not found in dump '{dump_name}'"
        )
        path_from_obj1 = explorer.find_path_between_objects(
            obj1_id, common_object_id, avoiding_ids=set()
        )
        path_from_obj2 = explorer.find_path_between_objects(
            obj2_id, common_object_id, avoiding_ids=set()
        )
        return render_template(
            "largest_common_reachable_object.html",
            dump_name=dump_name,
            obj1_id=obj1_id,
            obj2_id=obj2_id,
            common_object=common_object,
            path_from_obj1=path_from_obj1,
            path_from_obj2=path_from_obj2,
        )

    app.secret_key = os.urandom(
        16
    )  # This app keeps state in-process, so a per-process key is fine.
    return app


def main():
    """Run the local Midden web server."""
    arg_parser = argparse.ArgumentParser(description="Run the Midden web server")
    arg_parser.add_argument("--host", default="127.0.0.1")
    arg_parser.add_argument("--port", default=5000, type=int)
    arg_parser.add_argument(
        "--no-start-web-browser",
        action="store_false",
        dest="start_web_browser",
        help="Don't automatically open the web browser",
    )
    args = arg_parser.parse_args()
    app = create_app()
    url = f"http://{args.host}:{args.port}"
    print(f"Starting Midden web server on {url}")
    server = WSGIServer((args.host, args.port), app)
    if args.start_web_browser:
        webbrowser.open(url)
    server.start()


if __name__ == "__main__":
    main()
