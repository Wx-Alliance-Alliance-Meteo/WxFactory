from configparser import ConfigParser, NoSectionError, NoOptionError
import json
import types
from typing import TypeVar, Type, Union, Optional, Callable, Any, Literal, List

__all__ = ["ConfigurationField", "ConfigurationSchema", "OptionType", "load_default_schema"]


CaseSensitiveStr = types.new_class(name="cs-str", bases=(str,))

OptionType = TypeVar("OptionType", bound=Union[str, CaseSensitiveStr, int, float, List[int], List[float], bool])
_T = TypeVar("T", str, dict, list)
_Numerical = TypeVar("Numerical", bound=Union[int, float])
_Selectable = TypeVar("Selectable", bound=Union[int, float, str])


def _default_validate(_: Any) -> Literal[True]:
    return True


def make_str(items, markdown: bool = False, header: bool = False):
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

        if selectables is not None and not (min_value is None and max_value is None):
            raise ValueError(f"You cannot have both a min or a max, and a selectable pool of values")

    def make_validate(self) -> Callable:
        """Generate a function that checks whether a value (or list of values) falls within the range."""
        base_function = None
        if self.selectables is not None:
            base_function = lambda a: a in self.selectables

        elif self.min_value is not None and self.max_value is not None:
            base_function = lambda a: self.min_value <= a <= self.max_value
        elif self.min_value is not None:
            base_function = lambda a: a >= self.min_value
        elif self.max_value is not None:
            base_function = lambda a: a <= self.max_value

        if base_function is not None:

            def validate(a):
                if isinstance(a, list):
                    return all(base_function(x) for x in a)
                return base_function(a)

            return validate

        return lambda _: True

    def __str__(self):
        if self.selectables is not None:
            sel_str = ", ".join(str(s) for s in self.selectables)
            return f"{{{sel_str}}}"
        elif self.min_value is not None or self.max_value is not None:
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

    __name__ = "cs-str"

    def __init__(
        self,
        field_name: str,
        field_section: str,
        field_default: Optional[OptionType],
        field_type: Type[OptionType],
        valid_range: ConfigFieldRange,
        dependency: Optional[tuple[str, List[OptionType]]],
        description: str,
    ):
        self.name = field_name
        self.section = field_section
        self.field_default = field_default
        self.type = field_type
        self.valid_range = valid_range
        self.validate = self.valid_range.make_validate()
        self.dependency = dependency
        self.description = description

        if field_default is not None and not self.validate(self.field_default):
            raise ValueError(
                f"Default value '{self.field_default}' of field '{self.name}' does not respect "
                f"its valid range: {self.valid_range}"
            )

    def read(self, parser: ConfigParser) -> OptionType:
        try:
            if self.type == float:
                value = parser.getfloat(self.section, self.name)
            elif self.type == int:
                value = parser.getint(self.section, self.name)
            elif self.type == str:
                value = parser.get(self.section, self.name).lower()
            elif self.type == CaseSensitiveStr:
                value = parser.get(self.section, self.name)
            elif self.type == bool:
                value = bool(parser.getint(self.section, self.name))
            elif self.type == List[int]:
                try:
                    value = [parser.getint(self.section, self.name)]
                except ValueError:
                    value = [int(x) for x in json.loads(parser.get(self.section, self.name))]
            elif self.type == List[float]:
                try:
                    value = [parser.getfloat(self.section, self.name)]
                except ValueError:
                    value = [float(x) for x in json.loads(parser.get(self.section, self.name))]
            else:
                raise ValueError(f"Cannot get this option type (not implemented): {self.type}")

        except (NoOptionError, NoSectionError) as e:
            if self.field_default is None:
                raise ValueError(f"\nMust specify a value for option '{self.name}'") from e

            value = self.field_default

        if not self.validate(value):
            raise ValueError(
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


def check_list_type(values: list, expected_type: Type[_T], non_empty: bool = False):
    """Check that every element in the given list has the expected type.
    Raise a TypeError if that's not the case.
    If the list is 'None', it's always correct."""

    if values is not None and len(values) == 0 and non_empty:
        raise ValueError(f"List should not be empty! {values}")

    if values is not None and not all(isinstance(v, expected_type) for v in values):
        raise TypeError(f"Expected type list of {expected_type}, got {[type(v) for v in values]}")


class ConfigurationSchema:

    def __init__(self, json_str: str):
        try:
            format_obj = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("The configuration schema file is badly formatted. It must be a valid JSON file") from None

        self.version = self.__get_attribute("version", format_obj, str)
        sections = self.__get_attribute("sections", format_obj, list, optional=True)
        field_list = self.__get_attribute("fields", format_obj, list, optional=True)

        fields = []
        if sections is not None:
            for s in sections:
                fields.extend(self.__extract_section(s))

        if field_list is not None:
            fields.extend([self.__extract_field(f) for f in field_list])

        # Sort fields for dependency (also check if they make sense)
        self.fields: list[ConfigurationField] = []
        self.fields = fields

    def __get_attribute(
        self, attribute_name: str, attributes: dict[str, _T], return_type: Type[_T], optional: bool = False
    ) -> Optional[_T]:
        """Retrieve an attribute from the given dictionary and verify that it has the specified type."""
        try:
            attribute = attributes[attribute_name]
            if not isinstance(attribute, return_type):
                raise TypeError(f"Expected type {return_type} for '{attribute_name}', but got {type(attribute)}")
        except KeyError as e:
            if not optional:
                raise KeyError(f"'{attribute_name}' field not found in the dictionary {attributes}") from e
            return None

        return attribute

    def __extract_section(self, section: dict) -> list[ConfigurationField]:
        name = self.__get_attribute("name", section, str)
        field_list = self.__get_attribute("fields", section, list)
        fields = [self.__extract_field(f, name) for f in field_list]

        return fields

    def __get_range(self, field: dict, return_type: Type[_Numerical]) -> ConfigFieldRange:
        min_value = self.__get_attribute("min", field, return_type, optional=True)
        max_value = self.__get_attribute("max", field, return_type, optional=True)
        selectables = self.__get_attribute("selectables", field, list, optional=True)
        check_list_type(selectables, return_type, non_empty=True)

        return ConfigFieldRange(min_value, max_value, selectables)

    def __parse_numerical_field(
        self, field: dict, return_type: Type[_Numerical]
    ) -> tuple[Optional[_Numerical], Type[_Numerical], Callable[[_Numerical], bool]]:
        field_default = self.__get_attribute("default", field, return_type, optional=True)
        valid_range = self.__get_range(field, return_type)

        return field_default, return_type, valid_range

    def __parse_bool_field(
        self, field: dict
    ) -> tuple[Optional[bool], bool, Callable[[Any], bool], Callable[[int], bool]]:
        field_default = self.__get_attribute("default", field, bool, optional=True)

        def transform(value: int) -> bool:
            return bool(value)

        if field_default is not None:
            field_default = transform(field_default)

        return field_default, bool, transform

    def __parse_str_field(self, field: dict, case_sensitive: bool) -> tuple[Optional[str], str, Callable[[str], bool]]:
        field_default = self.__get_attribute("default", field, str, optional=True)
        valid_range = self.__get_range(field, str)
        if not case_sensitive:
            if field_default is not None:
                field_default = field_default.lower()
            if valid_range.selectables is not None:
                valid_range.selectables = [selectable.lower() for selectable in valid_range.selectables]

        def transform(value: str) -> str:
            if not case_sensitive:
                return value.lower()

            return value

        t = CaseSensitiveStr if case_sensitive else str

        return field_default, t, valid_range, transform

    def __parse_numerical_list_field(
        self, field: dict, return_type: Type[_Numerical]
    ) -> tuple[Optional[List[_Numerical]], Type[List[_Numerical]], Callable[[List[_Numerical]], bool]]:
        field_default = self.__get_attribute("default", field, list, optional=True)
        check_list_type(field_default, return_type)
        valid_range = self.__get_range(field, return_type)

        return field_default, List[return_type], valid_range

    def __extract_field(self, field: dict, field_section: str = "") -> ConfigurationField:
        field_name = self.__get_attribute("name", field, str)
        if field_section == "":
            field_section = self.__get_attribute("section", field, str)
        field_type_value = self.__get_attribute("type", field, str)
        field_default: Optional[OptionType] = None
        field_type: Type[OptionType]
        transform: Callable[[OptionType], OptionType] = lambda x: x

        try:
            match field_type_value:
                case "int":
                    field_default, field_type, valid_range = self.__parse_numerical_field(field, int)
                case "float":
                    field_default, field_type, valid_range = self.__parse_numerical_field(field, float)
                case "bool":
                    field_default, field_type, transform = self.__parse_bool_field(field)
                    valid_range = ConfigFieldRange()
                case "case-sensitive-str":
                    field_default, field_type, valid_range, transform = self.__parse_str_field(field, True)
                case "str":
                    field_default, field_type, valid_range, transform = self.__parse_str_field(field, False)
                case "list-int":
                    field_default, field_type, valid_range = self.__parse_numerical_list_field(field, int)
                case "list-float":
                    field_default, field_type, valid_range = self.__parse_numerical_list_field(field, float)
                case _:
                    raise ValueError(f"The type '{field_type_value}' of field '{field_name}' is not valid")

        except Exception as e:
            raise ValueError(f"Field {field_name}") from e

        dependency = None
        dep = self.__get_attribute("dependency", field, dict, optional=True)
        if dep is not None:
            dep_field = self.__get_attribute("name", dep, str)
            dep_values = self.__get_attribute("values", dep, list)
            dependency = dep_field, dep_values

        description = self.__get_attribute("description", field, str, optional=True)

        return ConfigurationField(
            field_name, field_section, field_default, field_type, valid_range, dependency, description
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
        return "\n".join(f"{f.name}: {f.type.__name__}" for f in self.fields)


def load_default_schema() -> ConfigurationSchema:
    import wx_mpi

    schema_path = "config/config-format.json"
    schema_content = wx_mpi.readfile(schema_path)
    schema = ConfigurationSchema(schema_content)

    return schema
