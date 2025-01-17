from common.configuration import Configuration

def pack(configuration: Configuration):
    sects = {}
    for section_name, section_options in configuration.sections.items():
        sects[section_name] = {}
        for option in section_options:
            val = getattr(configuration, option)
            sects[section_name][option] = val
    return sects
