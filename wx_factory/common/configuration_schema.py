from collections import OrderedDict
from configparser import ConfigParser, NoSectionError, NoOptionError
import copy
import json
import types
from typing import TypeVar, Type, Union, Optional, Callable, Any, Literal, List

import numpy

from .angle24 import angle24
from .eval_expr import eval_expr

__all__ = ["ConfigurationField", "ConfigurationSchema", "OptionType", "load_default_schema"]


class ConfigValueError(ValueError):
    """Custom exception for config schema/options"""


class CaseSensitiveStr(str):
    """Custom string option type"""


CaseSensitiveStr.__name__ = "cs-str"


class LowerCaseStr(str):
    """Custom string option type (always lower case)"""

    def __new__(cls, val: str):
        return super().__new__(cls, val.lower())


LowerCaseStr.__name__ = "lc-str"


def str_to_bool(val: str):
    return bool(int(val))


OptionType = TypeVar("OptionType", bound=Union[str, CaseSensitiveStr, int, float, List[int], List[float], bool])
_T = TypeVar("T", str, dict, list)
_Numerical = TypeVar("Numerical", bound=Union[int, float, angle24, numpy.float32])
_Selectable = TypeVar("Selectable", bound=Union[int, float, str])


default_schema_path = "config/config-format.json"

def needs_evaluation(attribute: _T, attribute_type: Type[_T]) -> bool:
    try:
        float(attribute)
        is_numeric = True
    except:
        is_numeric = False

    to_numeric = attribute_type in _Numerical.__bound__.__args__
    return not is_numeric and to_numeric and isinstance(attribute, str) and not issubclass(attribute_type, str)


def make_str(items, markdown: bool = False, header: bool = False):
    """Create a string from a set of items, each with their own fixed length. If an
    item is larger than its associated length, it will take space from the following items."""
    result = ""
    desired_length = 0
    for item in items:
        desired_length += item[1]
        if markdown:
            result += " | "
        if markdown and header:
            result += f"**{item[0]}** "
        else:
            result += item[0] + " "
        num_missing = desired_length - len(result)
        if num_missing > 0:
            result += f"{'':{num_missing}s}"

    if markdown:
        result += " | "

    return result


class ConfigFieldRange:
    """Define acceptable values for a certain field. Also generates a function to verify that a value falls within
    the range."""

    def __init__(
        self, min_value: OptionType = None, max_value: OptionType = None, selectables: List[OptionType] = None
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.selectables = selectables

        # Make sure that the range makes sense
        if selectables is not None and not (min_value is None and max_value is None):
            raise ConfigValueError(f"You cannot have both a min or a max, and a selectable pool of values")

        if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
            raise ConfigValueError(f"Min value {self.min_value} is larger than max {self.max_value}")

    def _make_base_validate(self):
        """Make a function that checks if a single value meets the requirements."""
        if self.selectables is not None:
            return lambda a: a in self.selectables
        if self.min_value is not None and self.max_value is not None:
            return lambda a: self.min_value <= a <= self.max_value
        if self.min_value is not None:
            return lambda a: a >= self.min_value
        if self.max_value is not None:
            return lambda a: a <= self.max_value

        return None

    def make_validate(self) -> Callable[[Any], bool]:
        """Generate a function that checks whether a value (or list of values) falls within the range."""
        base_function = self._make_base_validate()
        if base_function is None:
            return lambda _: True

        def validate(a) -> bool:
            """Verify that the given value falls within this range"""
            if isinstance(a, list):
                return all(base_function(x) for x in a)
            return base_function(a)

        return validate

    def __str__(self):
        if self.selectables is not None:
            sel_str = ", ".join(str(s) for s in self.selectables)
            return f"{{{sel_str}}}"

        if self.min_value is not None or self.max_value is not None:
            min_str = "-inf" if self.min_value is None else f"{self.min_value}"
            max_str = "inf" if self.max_value is None else f"{self.max_value}"
            return f"[{min_str}, {max_str}]"

        return ""


class ConfigurationField:
    """
    An option that can be set in a configuration file. Has a name, a type, a section and potentially
    a default value and range of possible values. Its default value can also depend on the value of
    another option.
    """

    def __init__(
        self,
        field_name: str,
        field_section: str,
        field_default: Optional[OptionType],
        field_type: Type[OptionType],
        is_list: bool,
        valid_range: ConfigFieldRange,
        dependency: Optional[tuple[str, List[OptionType]]],
        description: str,
    ):
        self.name = field_name
        self.section = field_section
        self.field_default = field_default
        self.type = field_type
        self.is_list = is_list
        self.valid_range = valid_range
        self.validate = self.valid_range.make_validate()
        self.dependency = dependency
        self.description = description

        if field_default is not None and not self.validate(self.field_default):
            raise ValueError(
                f"Default value '{self.field_default}' of field '{self.name}' does not respect "
                f"its valid range: {self.valid_range}"
            )

    def _read_single(self, parser: ConfigParser):
        """Read this field from the given parser (scalar)"""
        value = parser.get(self.section, self.name)
        return self.type(eval_expr(value) if needs_evaluation(value, self.type) else value)

    def _read_list(self, parser: ConfigParser):
        """Read this field from the given parser (list)"""
        try:
            return [self.type(parser.get(self.section, self.name))]
        except ValueError:
            return [self.type(eval_expr(x)) if needs_evaluation(x, self.type) else self.type(x) for x in json.loads(parser.get(self.section, self.name))]

    def typename(self, inner=False):
        if self.is_list and not inner:
            return f"List[{self.typename(True)}]"

        if self.type == angle24:
            return float.__name__
        elif self.type == LowerCaseStr:
            return str.__name__
        elif self.type == str_to_bool:
            return bool.__name__

        return self.type.__name__

    def read(self, parser: ConfigParser) -> OptionType:
        """Read this field from the given parser and verify that it falls within its valid range."""

        try:
            if self.is_list:
                value = self._read_list(parser)
            else:
                value = self._read_single(parser)

        except (NoOptionError, NoSectionError) as e:
            if self.field_default is None:
                raise ConfigValueError(f"\nMust specify a value for option '{self.name}'") from e

            value = copy.deepcopy(self.field_default)

        except ValueError as e:
            raise ConfigValueError(e) from e

        if not self.validate(value):
            raise ConfigValueError(
                f"Value '{value}' does not fall in acceptable range for field '{self.name}': {self.valid_range}"
            )

        return value

    @staticmethod
    def header(section_name: str, markdown: bool = False) -> str:
        """Create a section header"""
        header_str = make_str(
            [(f"[{section_name}]", 24), ("Type", 8), ("Default", 12), ("Valid range", 35), ("Description", 0)],
            markdown,
            header=True,
        )

        if markdown:
            header_str = "| | | | | |\n" + header_str

        return header_str

    def to_string(self, markdown: bool = False) -> str:
        """Make a summary of this field that will fit nicely in a table."""
        return make_str(
            [
                (f"{self.name}", 24),
                (f"{self.type.__name__}", 8),
                (f"{'[none]' if self.field_default is None else str(self.field_default)}", 12),
                (f"{'' if self.valid_range is None else str(self.valid_range)}", 35),
                (f"{'' if self.description is None else self.description}", 0),
            ],
            markdown,
        )

    def __str__(self):
        return self.to_string(markdown=False)


# class LastUpdatedOrderedDict(OrderedDict):
#     "Store items in the order the keys were last added"

#     def __setitem__(self, key, value):
#         super().__setitem__(key, value)
#         self.move_to_end(key)


def sort_fields_by_dependency(fields: list[ConfigurationField]) -> list[ConfigurationField]:
    # fields_dict: dict[ConfigurationField] = LastUpdatedOrderedDict()
    sorted_fields: list[ConfigurationField] = []
    remainder: list[ConfigurationField] = []
    for f in fields:
        # if f.name in fields_dict:
        #     raise ConfigValueError(f"Duplicate field name {f.name}")
        if f.dependency is None:
            sorted_fields.append(f)
            # fields_dict[f.name] = f
        else:
            remainder.append(f)

    for f in remainder:
        # if f.dependency[0] not in fields_dict:
        #     raise ConfigValueError(f"Field {f.name} depends on a field that does not exist {f.dependency[0]}")
        # fields_dict[f.name] = f
        sorted_fields.append(f)

    return sorted_fields
    # return [f for _, f in fields_dict.items()]


class ConfigurationSchema:
    """A description of the options that can be configured in a config file. This descriptions includes
    option names, their default value, the range of acceptable values and some basic dependency between
    options.

    The schema is loaded from a JSON description. Each option is associated with a section; the JSON file
    may have a list of sections, each containig a list of fields, or it can have a single list of fields,
    each with an associated section name.
    """

    def __init__(self, json_str: str):
        try:
            format_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError("The configuration schema file is badly formatted. It must be a valid JSON file") from e

        self.raw_string = json_str

        self.version = self.__get_attribute("version", format_obj, str)
        sections = self.__get_attribute("sections", format_obj, dict, optional=True, is_list=True)
        field_list = self.__get_attribute("fields", format_obj, dict, optional=True, is_list=True)

        fields: list[ConfigurationField] = []
        if sections is not None:
            for s in sections:
                fields.extend(self.__extract_section(s))

        if field_list is not None:
            fields.extend([self.__extract_field(f) for f in field_list])

        self.fields = sort_fields_by_dependency(fields)

    def __get_attribute(
        self,
        attribute_name: str,
        attributes: dict[str, _T],
        attribute_type: Type[_T],
        optional: bool = False,
        is_list: bool = False,
    ) -> Optional[_T]:
        """Retrieve an attribute from the given dictionary as the specified type. Raise an exception if
        the attribute does not exist (if not optional) or if it cannot be converted to the specified type."""

        if attribute_name not in attributes:
            if not optional:
                raise KeyError(f"'{attribute_name}' field not found in the dictionary {attributes}")
            return None

        attribute = attributes[attribute_name]

        # Evaluate expression from string, if appropriate
        if needs_evaluation(attribute, attribute_type):
            return attribute_type(eval_expr(attribute))

        # Convert to list, if necessary
        if is_list:
            if not isinstance(attribute, list):
                attribute = [attribute]
            if issubclass(attribute_type, list):
                return attribute
            return [attribute_type(eval_expr(a) if needs_evaluation(a, attribute_type) else a) for a in attribute]
        
        return attribute_type(attribute)

    def __extract_section(self, section: dict) -> list[ConfigurationField]:
        """Extract all fields from the given section"""
        section_name = self.__get_attribute("name", section, str)
        field_list = self.__get_attribute("fields", section, dict, is_list=True)
        fields = [self.__extract_field(f, section_name) for f in field_list]

        return fields

    def __get_range(self, field: dict, return_type: Type[_Numerical]) -> ConfigFieldRange:
        """Extract the valid range of values for the given field."""
        min_value = self.__get_attribute("min", field, return_type, optional=True)
        max_value = self.__get_attribute("max", field, return_type, optional=True)
        selectables = self.__get_attribute("selectables", field, return_type, optional=True, is_list=True)

        return ConfigFieldRange(min_value, max_value, selectables)

    def __extract_field(self, field: dict, field_section: str = "") -> ConfigurationField:
        """Extract a single field from the given set."""
        field_name = self.__get_attribute("name", field, str)
        if field_section == "":
            field_section = self.__get_attribute("section", field, str)
        field_type_value = self.__get_attribute("type", field, str)

        # Available types (some with multiple aliases)
        classes = {
            "int": int,
            "float": float,
            "float32": numpy.float32,
            "float64": float,
            "angle24": angle24,
            "bool": str_to_bool,
            "case-sensitive-str": CaseSensitiveStr,
            "cs-str": CaseSensitiveStr,
            "lower-case-str": LowerCaseStr,
            "lc-str": LowerCaseStr,
            "str": LowerCaseStr,
        }
        if field_type_value[:5] == "list-":
            is_list = True
            field_type = classes[field_type_value[5:]]
        else:
            is_list = False
            field_type = classes[field_type_value]
        try:
            field_default = self.__get_attribute("default", field, field_type, optional=True, is_list=is_list)
            valid_range = self.__get_range(field, field_type)

            dependency = None
            dep = self.__get_attribute("dependency", field, dict, optional=True)
            if dep is not None:
                dep_field = self.__get_attribute("name", dep, str)
                dep_values = self.__get_attribute("values", dep, list, is_list=True)
                dependency = dep_field, dep_values

            description = self.__get_attribute("description", field, str, optional=True)

        except Exception as e:
            raise ValueError(f"Field {field_name}") from e

        return ConfigurationField(
            field_name, field_section, field_default, field_type, is_list, valid_range, dependency, description
        )

    def to_string(self, markdown: bool = False):
        sections = {}
        for field in self.fields:
            if field.section in sections:
                sections[field.section].append(field)
            else:
                sections[field.section] = [field]

        section_strings = [
            f"{ConfigurationField.header(title, markdown)}\n  " + "\n  ".join([f"{f.to_string(markdown)}" for f in s])
            for title, s in sections.items()
        ]
        if markdown:
            section_strings = ["| | | | | |", "| - | - | - | - | - |"] + section_strings
        return "\n".join(section_strings)

    def __str__(self):
        return self.to_string(markdown=False)

    def type_hints(self):
        return "\n".join(f"{f.name}: {f.typename()}" for f in self.fields)


def load_default_schema() -> ConfigurationSchema:

    with open(default_schema_path) as f:
        schema_content = "\n".join(f.readlines())

    return ConfigurationSchema(schema_content)
