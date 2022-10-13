from .equations import library
from .compartment import Compartment, Soma, Dendrite
from .neuronmodel import NeuronModel
from .ephysproperties import EphysProperties

def check_dependencies():
    """
    Check if Brian 2 is installed
    """
    import sys

    try:
        import brian2
    except ImportError as ex:
        sys.stderr.write(f"Importing brian2 failed: '{ex}'\n")
        raise ImportError(
            f"Please install Brian 2 to use dendrify")


check_dependencies()
