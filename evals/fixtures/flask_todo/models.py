from dataclasses import asdict, dataclass


@dataclass
class Todo:
    title: str
    id: int = 0
    done: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
