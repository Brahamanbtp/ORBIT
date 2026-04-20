class ActionSpace:
    def __init__(self, action_names: list[str]) -> None:
        self.action_names = action_names

    def get_name(self, action_id: int) -> str:
        return self.action_names[action_id]

    def size(self) -> int:
        return len(self.action_names)
