from .a2c import A2C

###


class MetaA2C(A2C):

    @property
    def name(self) -> str:
        return "MetaA2C"

    @property
    def meta_algorithm(self) -> bool:
        return True
