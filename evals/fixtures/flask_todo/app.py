"""Tiny todo API used as an evaluation fixture."""
from flask import Flask, jsonify, request

from models import Todo
from storage import TodoStore

app = Flask(__name__)
store = TodoStore("todos.json")


@app.route("/todos", methods=["GET"])
def list_todos():
    return jsonify([t.to_dict() for t in store.all()])


@app.route("/todos", methods=["POST"])
def create_todo():
    payload = request.get_json(force=True)
    todo = Todo(title=payload["title"])
    store.add(todo)
    return jsonify(todo.to_dict()), 201


@app.route("/todos/<int:todo_id>/done", methods=["POST"])
def mark_done(todo_id):
    # TODO: return 404 when the id does not exist
    store.complete(todo_id)
    return "", 204


if __name__ == "__main__":
    app.run(port=5000)
