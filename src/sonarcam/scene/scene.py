from .objects import SceneObject
class Scene:
    def __init__(self):
        self.objects = [SceneObject()]  # one default cube
    def add_object(self, obj: SceneObject): self.objects.append(obj)
    def remove_index(self, i: int):
        if 0 <= i < len(self.objects): self.objects.pop(i)
