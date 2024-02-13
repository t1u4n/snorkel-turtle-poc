from typing import Callable, List, Any

class LabelingFunctionLibrary:
    """A library of labeling functions that can be used to assign labels to data points.

    A labeling function is a callable object that takes as input an arbitrary number of
    arguments and returns a single value (the label).
    """

    def __init__(self) -> None:
        """Initialize an empty labeling function library."""
        self.label_functions = {}

    def register(self, name: str, func: Callable[[Any], Any]) -> None:
        """Register a new label function with the library."""
        if not callable(func):
            raise TypeError("The provided argument is not a function.")
        elif name in self.label_functions:
            raise ValueError(f"A label function named '{name}' has already been registered.")
        else:
            self.label_functions[name] = func

    def get(self, name: str) -> Callable[[Any], Any]:
        """Get a previously-registered label function by its name."""
        try:
            return self.label_functions[name]
        except KeyError as e:
            raise LookupError(f"No such label function: {e}") from e
    
    def get_all(self) -> List[Callable[[Any], Any]]:
        """Return all of the registered label functions."""  # TODO: This should probably be sorted
        return list(self.label_functions.values())
    
    def unregister(self, name: str) -> None:
        """Remove a label function from the library."""
        del self.label_functions[name]
    
    def clear(self) -> None:
        """Remove all label functions from the library."""
        self.label_functions = {}