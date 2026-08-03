"""Persistent, validated color-profile loading for lane detection nodes."""

from copy import deepcopy
from pathlib import Path

import yaml


ACTIVE_PROFILE_PATH = Path.home() / '.config' / 'lane_offset' / 'active_color_profile'
REQUIRED_COLORS = {
    'mission': {'white', 'green', 'light_gray'},
    'timed': {'white', 'green', 'light_gray', 'dark_gray'},
}


def profile_file_path():
    """Return installed config path, with a source-tree fallback for development."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory('lane_offset')) / 'config' / 'color_profiles.yaml'
    except (ImportError, LookupError):
        return Path(__file__).resolve().parents[1] / 'config' / 'color_profiles.yaml'


def read_profiles(path=None):
    path = Path(path) if path else profile_file_path()
    with path.open(encoding='utf-8') as stream:
        try:
            data = yaml.safe_load(stream) or {}
        except yaml.YAMLError as error:
            raise ValueError(f'Invalid YAML in {path}: {error}') from error
    profiles = data.get('profiles')
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f'No profiles found in {path}')
    default = data.get('default_profile')
    if default not in profiles:
        raise ValueError(f'default_profile {default!r} does not exist in {path}')
    return data, path


def selected_profile_name(data):
    try:
        selected = ACTIVE_PROFILE_PATH.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        selected = ''
    return selected if selected in data['profiles'] else data['default_profile']


def select_profile(name, path=None):
    data, _ = read_profiles(path)
    if name not in data['profiles']:
        raise ValueError(f'Unknown color profile: {name}')
    for section in ('mission', 'timed'):
        colors = _validated_colors(data['profiles'][name].get(section), section)
        if set(colors) != REQUIRED_COLORS[section]:
            raise ValueError(
                f'Profile {name!r}/{section} colors must be '
                f'{sorted(REQUIRED_COLORS[section])}, got {sorted(colors)}'
            )
    ACTIVE_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_PROFILE_PATH.write_text(name + '\n', encoding='utf-8')


def load_color_classes(default_classes, section, path=None):
    """Overlay the selected profile's thresholds on class metadata from a node."""
    data, source = read_profiles(path)
    name = selected_profile_name(data)
    overrides = _validated_colors(data['profiles'][name].get(section), section)
    result = deepcopy(default_classes)
    expected = {item['name'] for item in result}
    if set(overrides) != expected:
        raise ValueError(
            f'Profile {name!r}/{section} colors must be {sorted(expected)}, '
            f'got {sorted(overrides)}'
        )
    for color_class in result:
        values = overrides[color_class['name']]
        color_class['hsv'] = values['hsv']
        color_class['ycrcb'] = values.get('ycrcb')
    return result, name, source


def _validated_colors(colors, section):
    if not isinstance(colors, dict) or not colors:
        raise ValueError(f'Missing or empty {section!r} profile section')
    result = {}
    for name, values in colors.items():
        if not isinstance(values, dict):
            raise ValueError(f'{section}.{name} must be a mapping')
        normalized = {'hsv': _validate_ranges(values.get('hsv'), ('h', 's', 'v'), name)}
        ycrcb = values.get('ycrcb')
        normalized['ycrcb'] = (
            _validate_ranges(ycrcb, ('y', 'cr', 'cb'), name) if ycrcb is not None else None
        )
        result[name] = normalized
    return result


def _validate_ranges(values, channels, color_name):
    if not isinstance(values, dict) or set(values) != set(channels):
        raise ValueError(f'{color_name} must define exactly {channels}')
    limits = {'h': 179, 's': 255, 'v': 255, 'y': 255, 'cr': 255, 'cb': 255}
    result = {}
    for channel in channels:
        bounds = values[channel]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f'{color_name}.{channel} must contain [min, max]')
        low, high = bounds
        if not isinstance(low, int) or not isinstance(high, int):
            raise ValueError(f'{color_name}.{channel} bounds must be integers')
        if not 0 <= low <= high <= limits[channel]:
            raise ValueError(f'Invalid {color_name}.{channel} range: {bounds}')
        result[channel] = (low, high)
    return result
