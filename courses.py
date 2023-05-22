import json
import random


class CoursePromptGenerator:
    def generate_exercise(self) -> str:
        return (
            "Utilsant ma réponse ci-dessus et les points ci-dessous, "
            "rédigez une réponse digne d'un professeur (détaillée et argumentée). "
            "Puis créez un exercise ou une question pour que je puisse pratiquer la langue Corse."
        )

    def generate_user(self) -> str:
        return self.generate_exercise()


class ConjugaisonCoursePromptGenerator(CoursePromptGenerator):
    # TODO: grab a random verb to request a conjugation exercise on it.
    def __init__(self) -> None:
        super().__init__()
        self.load_verbs()

    def load_verbs(self):
        with open("./documents/corsica_verbs.json") as f:
            self.alphabetic_verbs: dict[str, list[str]] = json.load(f)
            self.all_verbs = [verb for verbs in self.alphabetic_verbs.values() for verb in verbs]

    def get_random_verb(self):
        return random.choice(self.all_verbs)

    def generate_exercise(self):
        self.verb = self.get_random_verb()
        return (
            f"{super().generate_exercise()}\n"
            f"- Apprendre le verbe: {self.verb}\n"
            "- Definition du verbe en Corse et en Français\n"
            "- Utilisation du verbe dans une phrase en Corse\n"
            "- Question pour confirmer la comprehension du verbe dans une phrase en Corse"
        )

    def generate_user(self, response):
        return f"{response}: le verbe {self.verb}"


COURSES_FACTORY = {
    "default": CoursePromptGenerator(),
    "exercice de conjugaison": ConjugaisonCoursePromptGenerator(),
    # "exercice d'ortographe": OrtographCoursePromptGenerator(),
}


class CourseManager:
    """Manage the user course"""

    def __init__(self, client_id) -> None:
        self.client_id = client_id

    def update_inputs(self, response: str, chat_history: list):
        content_question = ""
        if not chat_history:
            if response in COURSES_FACTORY:
                content_question = COURSES_FACTORY[response].generate_exercise()
        else:
            pre_content = COURSES_FACTORY["default"].generate_exercise()
            content_question = response
            response += f"\n{pre_content}"

        return response, content_question
