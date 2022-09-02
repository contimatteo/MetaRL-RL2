from .a3c import A3C

###


class MetaA3C(A3C):

    @property
    def name(self) -> str:
        return "MetaA3C"

    @property
    def meta_algorithm(self) -> bool:
        return True
