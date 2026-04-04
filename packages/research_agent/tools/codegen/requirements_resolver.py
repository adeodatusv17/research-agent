class RequirementsResolver:
    def resolve(self, framework: str = "pytorch") -> list[str]:
        return ["torch", "pyyaml"]
