"""
Methods for searching for conjunctions in a binary (formal) search context.
"""

from .context import SearchContext
from .core_query_tree import CoreQueryTreeSearch
from .greedy import GreedySearch

#: Dictionary of available search methods.
search_methods = {
    'exhaustive': CoreQueryTreeSearch,
    'greedy': GreedySearch,
}

__all__ = [
    'SearchContext',
    'search_methods',
]