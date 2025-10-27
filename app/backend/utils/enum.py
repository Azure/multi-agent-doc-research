from enum import Enum, EnumMeta
from typing import Any


class CaseInsensitiveEnumMeta(EnumMeta):
    """Enum metaclass to allow for interoperability with case-insensitive strings.

    Consuming this metaclass in an SDK should be done in the following manner:

    .. code-block:: python

        from enum import Enum
        from azure.core import CaseInsensitiveEnumMeta

        class MyCustomEnum(str, Enum, metaclass=CaseInsensitiveEnumMeta):
            FOO = 'foo'
            BAR = 'bar'

    """

    def __getitem__(cls, name: str) -> Any:
        # disabling pylint bc of pylint bug https://github.com/PyCQA/astroid/issues/713
        return super(CaseInsensitiveEnumMeta, cls).__getitem__(name.upper())

    def __getattr__(cls, name: str) -> Enum:
        """Return the enum member matching `name`.

        We use __getattr__ instead of descriptors or inserting into the enum
        class' __dict__ in order to support `name` and `value` being both
        properties for enum members (which live in the class' __dict__) and
        enum members themselves.

        :param str name: The name of the enum member to retrieve.
        :rtype: ~azure.core.CaseInsensitiveEnumMeta
        :return: The enum member matching `name`.
        :raises AttributeError: If `name` is not a valid enum member.
        """
        try:
            return cls._member_map_[name.upper()]
        except KeyError as err:
            raise AttributeError(name) from err


class SearchEngine(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The possible values for search engines."""

    BING_GROUNDING = "grounding_bing"
    """Bing Grounding with Bing Custom Search."""
    BING_GROUNDING_CRAWLING = "grounding_bing_crawling"
    """Bing Grounding with Bing Custom Search & Crawling."""
    BING_SEARCH_CRAWLING = "bing_search_crawling"
    """The Bing search engine."""
    GOOGLE_SEARCH_CRAWLING = "google_search_crawling"
    """The Google search engine."""
