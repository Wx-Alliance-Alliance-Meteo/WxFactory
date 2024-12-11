import json
from typing import TypeVar, Type, Union, Optional, Callable, Any, Literal, List
from common.configuration_types import OptionType

__all__ = ["ConfigurationField", "ConfigurationSchema"]

_T = TypeVar("T", str, dict, list)
_Numerical = TypeVar("Numerical", bound=Union[int, float])
_Selectable = TypeVar("Selectable", bound=Union[int, float, str])

def _default_validate(_: Any) -> Literal[True]:
    return True


class ConfigurationField:
    def __init__(
        self,
        field_name: str,
        field_section: str,
        field_default: Optional[OptionType],
        field_type: Type[OptionType],
        validate: Callable[[OptionType], bool],
        transform: Callable[[OptionType], OptionType],
        dependancy: Optional[tuple[str, List[OptionType]]]
    ):
        self.field_name = field_name
        self.field_section = field_section
        self.field_default = field_default
        self.field_type = field_type
        self.validate = validate
        self.transform = transform
        self.dependancy = dependancy


class ConfigurationSchema:

    def __get_attribute(
        self, attribute_name: str, object: dict[str, _T], attribute_err: str, type_err: str, return_type: Type[_T]
    ) -> _T:
        try:
            attribute = object[attribute_name]
            if type(attribute) != return_type:
                raise TypeError(type_err)
        except KeyError:
            raise ValueError(attribute_err) from None
        return attribute

    def __get_at_index(
        self, index: int, object: list[_T], attribute_err: str, type_err: str, return_type: Type[_T]
    ) -> _T:
        try:
            attribute = object[index]
            if type(attribute) != return_type:
                raise TypeError(type_err)
        except IndexError:
            raise ValueError(attribute_err) from None
        return attribute

    def __get_field_default(self, field: dict, type_err: str, return_type: Type[OptionType]) -> Optional[OptionType]:
        try:
            value = field["default"]

            if type(value) != return_type:
                raise TypeError(type_err)
        except KeyError:
            return None
        return value

    def __get_field_default_numerical_list(
        self, field: dict, type_err: str, return_type: Type[_Numerical]
    ) -> Optional[list[_Numerical]]:
        try:
            value = field["default"]

            if type(value) != list:
                raise TypeError(type_err)

            for it in range(len(value)):
                if type(value[it]) != return_type:
                    raise TypeError(type_err)
        except KeyError:
            return None
        return value

    def __get_min_max(
        self, field: dict, index: int, return_type: Type[_Numerical]
    ) -> tuple[Optional[_Numerical], Optional[_Numerical]]:
        min_value: Optional[_Numerical]
        max_value: Optional[_Numerical]

        try:
            min_value = field["min"]
            if type(min_value) != return_type:
                raise TypeError(f"Min value at field index {index} is not of the right type")
        except KeyError:
            min_value = None

        try:
            max_value = field["max"]
            if type(max_value) != return_type:
                raise TypeError(f"Max value at field index {index} is not of the right type")
        except KeyError:
            max_value = None

        return min_value, max_value

    def __get_selectables(
        self, field: dict, index: int, field_selectable_err: str, return_type: Type[_Selectable]
    ) -> Optional[list[_Selectable]]:
        selectables: list[_Selectable]

        try:
            selectables = field["selectables"]
            if type(selectables) != list:
                raise ValueError(f'"Selectable" field at field index {index} is not a list')
        except KeyError:
            return None

        if len(selectables) == 0:
            raise ValueError(f'"Selectable field at field index {index} should not be empty')

        for it in range(len(selectables)):
            if type(selectables[it]) != return_type:
                raise ValueError(field_selectable_err)

        return selectables

    def __parse_numerical_field(
        self, field: dict, index: int, field_default_err: str, field_selectable_err: str, return_type: Type[_Numerical]
    ) -> tuple[Optional[_Numerical], Type[_Numerical], Callable[[_Numerical], bool]]:
        field_default = self.__get_field_default(field, field_default_err, return_type)
        field_min, field_max = self.__get_min_max(field, index, return_type)
        field_selectables = self.__get_selectables(field, index, field_selectable_err, return_type)

        if field_selectables is not None and not (field_min is None and field_max is None):
            raise ValueError(
                f"You cannot have both a min or a max, and a selectable poll of value at field index {index}"
            )

        def validate(number: _Numerical) -> bool:
            valid: bool = True
            if field_selectables is not None:
                valid = valid and number in field_selectables
            if field_min is not None:
                valid = valid and number >= field_min
            if field_max is not None:
                valid = valid and number <= field_max
            return valid

        return field_default, return_type, validate

    def __parse_bool_field(self, field: dict, index: int) -> tuple[Optional[bool], bool, Callable[[Any], bool], Callable[[int], bool]]:
        field_default = self.__get_field_default(
            field, f'The "default" field at field index {index} is not a bool', bool
        )
        if field_default is not None:
            field_default = field_default > 0
        
        def transform(value: int) -> bool:
            return value != 0

        return field_default, bool, _default_validate, transform

    def __parse_str_field(
        self, field: dict, index: int, case_sensitive: bool
    ) -> tuple[Optional[str], str, Callable[[str], bool]]:
        field_default = self.__get_field_default(
            field, f'The "default" field at field index {index} is not a string', str
        )

        field_selectables = self.__get_selectables(
            field, index, f"One of the selectable item at field index {index} is not a string", str
        )
        if not case_sensitive:
            if field_default is not None:
                field_default = field_default.lower()
            if field_selectables is not None:
                field_selectables = [selectable.lower() for selectable in field_selectables]

        def validate(value: str) -> bool:
            valid: bool = True

            if field_selectables is not None:
                valid = valid and value in field_selectables

            return valid

        def transform(value: str) -> str:
            if not case_sensitive:
                return value.lower()

            return value

        return field_default, str, validate, transform

    def __parse_numerical_list_field(
        self, field: dict, index: int, field_default_err: str, field_selectable_err: str, return_type: Type[_Numerical]
    ) -> tuple[Optional[List[_Numerical]], Type[List[_Numerical]], Callable[[List[_Numerical]], bool]]:
        field_default = self.__get_field_default_numerical_list(field, field_default_err, return_type)
        field_min, field_max = self.__get_min_max(field, index, return_type)
        field_selectables = self.__get_selectables(field, index, field_selectable_err, return_type)

        if field_selectables is not None and not (field_min is None and field_max is None):
            raise ValueError(
                f"You cannot have both a min or a max, and a selectable poll of value at field index {index}"
            )

        def validate(numbers: list[_Numerical]) -> bool:
            valid: bool = True

            for number in numbers:
                if field_selectables is not None:
                    valid = valid and number in field_selectables
                if field_min is not None:
                    valid = valid and number >= field_min
                if field_max is not None:
                    valid = valid and number <= field_max
            return valid

        return field_default, List[return_type], validate

    def __get_field(self, field: dict, index: int) -> ConfigurationField:
        field_name = self.__get_attribute(
            "name",
            field,
            f'"Name" field not found in field index {index}',
            f'The "name" field at field index {index} is not a string',
            str,
        )
        field_section = self.__get_attribute(
            "section",
            field,
            f'"Section" field not found in field index {index}',
            f'The "section" field at field index {index} is not a string',
            str,
        )
        field_type_value = self.__get_attribute(
            "type",
            field,
            f'"Type" field not found in field index {index}',
            f'The "type" field at index {index} is not a string',
            str,
        )
        field_default: Optional[OptionType] = None
        field_type: Type[OptionType]
        validate: Callable[[OptionType], bool]
        transform: Callable[[OptionType], OptionType] = lambda x: x

        match field_type_value:
            case "int":
                field_default, field_type, validate = self.__parse_numerical_field(
                    field,
                    index,
                    f'The "default" field at field index {index} is not an int',
                    f"One of the selectable item at field index {index} is not an int",
                    int,
                )
            case "float":
                field_default, field_type, validate = self.__parse_numerical_field(
                    field,
                    index,
                    f'The "default" field at field index {index} is not a float',
                    f"One of the selectable item at field index {index} is not a float",
                    float,
                )

            case "bool":
                field_default, field_type, validate, transform = self.__parse_bool_field(field, index)

            case "case-sensitive-str":
                field_default, field_type, validate, transform = self.__parse_str_field(field, index, True)

            case "str":
                field_default, field_type, validate, transform = self.__parse_str_field(field, index, False)

            case "list-int":
                field_default, field_type, validate = self.__parse_numerical_list_field(
                    field,
                    index,
                    f'The "default" field at field index {index} is not an int',
                    f"One of the selectable item at field index {index} is not an int",
                    int,
                )

            case "list-float":
                field_default, field_type, validate = self.__parse_numerical_list_field(
                    field,
                    index,
                    f'The "default" field at field index {index} is not a float',
                    f"One of the selectable item at field index {index} is not a float",
                    float,
                )

            case _:
                raise ValueError(f'The "type" field at field index {index} is not of valid value')

        if field_default is not None:
            if not validate(field_default):
                raise ValueError(
                    f'The "default" field\'s value at field index {index} is does not respect its own limitation'
                )
        field_dependancy: Optional[tuple[str, List[OptionType]]]

        try:
            dep = self.__get_attribute(
                "dependancy", field, "", f'The "dependancy" field at field index {index} is not an object', dict
            )
        except ValueError:
            dependancy = None
        else:
            dep_field = self.__get_attribute(
                "name",
                dep,
                f'"Name" field in dependancy at field index {index} is required',
                f'The "name" field in dependancy at field index {index} is not a string',
                str,
            )
            dep_values = self.__get_attribute(
                "values",
                dep,
                f'"Values" field in dependancy at field index {index} is required',
                f'The "values" field in dependancy at field index {index} is not a list',
                list,
            )
            dependancy = dep_field, dep_values

        return ConfigurationField(field_name, field_section, field_default, field_type, validate, transform, dependancy)

    def __init__(self, json_str: str):
        try:
            format_obj = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("The configuration schema file is badly formatted. It must be a valid JSON file") from None

        self.version = self.__get_attribute(
            "version", format_obj, '"Version" field not found', 'The "version" field is not a string', str
        )
        fields_obj = self.__get_attribute(
            "fields", format_obj, '"Field" field not found', 'The "field" field is not a list', list
        )

        self.fields: list[ConfigurationField] = []
        for index in range(len(fields_obj)):
            field = self.__get_at_index(
                index, fields_obj, "Index out of bound", f"The field at {index} is not an object", dict
            )
            field = self.__get_field(field, index)
            self.fields.append(field)

def load_default_schema() -> ConfigurationSchema:
    schema_path = "config/config-format.json"
    schema_text: str
    with open(schema_path) as f:
        schema_text = "\n".join(f.readlines())
    schema = ConfigurationSchema(schema_text)
    return schema
