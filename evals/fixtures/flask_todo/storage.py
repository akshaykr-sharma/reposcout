import json
from pathlib import Path

from models import Todo


class TodoStore:
    """Keeps todos in a JSON file on disk."""

    def __init__(self, path: str):
        self.path = Path(path)

    def all(self) -> list[Todo]:
        if not self.path.exists():
            return []
        return [Todo(**row) for row in json.loads(self.path.read_text())]

    def add(self, todo: Todo) -> None:
        todos = self.all()
        todo.id = len(todos) + 1
        todos.append(todo)
        self._save(todos)

    def complete(self, todo_id: int) -> None:
        todos = self.all()
        for todo in todos:
            if todo.id == todo_id:
                todo.done = True
        self._save(todos)

    def _save(self, todos: list[Todo]) -> None:
        self.path.write_text(json.dumps([t.to_dict() for t in todos]))
