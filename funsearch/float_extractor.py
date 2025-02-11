import re
FLOAT = re.compile(r"(?<![^ ])[-+]?\d*\.\d+|(?<![^ ])[-+]?\d+")

class ProgramWrapper:
    def __init__(self, program: str):
        self.program = program
        self.floats = self.extract_floats()
        self.num_floats = len(self.floats)

    def extract_floats(self) -> list[float]:
        """Extracts all floats from a string."""
        return [float(x) for x in re.findall(FLOAT, self.program)]
    
    def sub_floats(self, numbers: list[float]) -> str:
        """Substitutes all floats in a string with the floats contained in the list "numbers"."""
        replacement = [str(x) for x in numbers]
        replacement_iter = iter(replacement)

        def replace_with_next(match: re.Match) -> str:
            return next(replacement_iter)

        self.program = re.sub(FLOAT, replace_with_next, self.program)
        self.floats = numbers
        self.num_floats = len(numbers)

    def get_program(self) -> str:
        return self.program
    
    def get_floats(self) -> list[float]:
        return self.floats
    
    def get_num_floats(self) -> int:
        return self.num_floats

if __name__ == "__main__":
    prompt = ("def heuristic(obs: np.ndarray) -> float:\n\n"
            "\"\"\"Returns an action between -1 and 1.\n"
            "obs size is 3.\n"
            "\"\"\"\n"
            "\tx1 = np.arctan2(-obs[1], obs[0])\n"
            "\tx2 = obs[2]\n"
            "\tif x1 < 0 and x2 > 0:\n"
            "\t\taction += 1\n"
            "\telif x1 > 0 and x2 < 0:\n"
            "\t\taction -= 1\n"
            "\telif abs(x1) >= abs(x2):\n"
            "\t\taction = np.sign(x1)\n"
            "\telse:\n"
            "\t\taction = np.sign(x2)\n\n"
            "\treturn action")
    p = ProgramWrapper(prompt)
    print(p.get_program())
    print(p.get_floats())
    replacement = [x for x in range(p.get_num_floats())]
    p.sub_floats(replacement)
    print(p.get_program())
    # for token in prompt.split():
    #     try:
    #         print(float(token), "is a float")
    #     except ValueError:
    #         print(token, "is something else")