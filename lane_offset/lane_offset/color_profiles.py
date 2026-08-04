"""Persistent, validated venue color palettes for lane detection nodes."""

from copy import deepcopy
from pathlib import Path

import yaml


ACTIVE_PROFILE_PATH = Path.home() / '.config' / 'lane_offset' / 'active_color_profile'
REQUIRED_COLORS = {'white', 'green', 'light_gray', 'dark_gray'}


def profile_file_path():
    """Return source config when available, then fall back to installed config."""
    module_path = Path(__file__).resolve()
    source_path = module_path.parents[1] / 'config' / 'color_profiles.yaml'
    source_candidates = [source_path]
    source_candidates.extend(
        parent.parent / 'lane_offset' / 'config' / 'color_profiles.yaml'
        for parent in module_path.parents
        if parent.name == 'install'
    )
    for candidate in source_candidates:
        if candidate.is_file():
            return candidate

    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory('lane_offset')) / 'config' / 'color_profiles.yaml'
    except (ImportError, LookupError):
        return source_path


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
    colors = _validated_colors(data['profiles'][name].get('colors'), name)
    missing = REQUIRED_COLORS - set(colors)
    if missing:
        raise ValueError(
            f'Profile {name!r} is missing required palette colors: '
            f'{sorted(missing)}'
        )
    ACTIVE_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_PROFILE_PATH.write_text(name + '\n', encoding='utf-8')


def load_color_classes(class_metadata, path=None):
    """Attach the active venue's thresholds to node-owned class metadata."""
    data, source = read_profiles(path)
    name = selected_profile_name(data)
    palette = _validated_colors(data['profiles'][name].get('colors'), name)
    result = deepcopy(class_metadata)
    expected = {item['name'] for item in result}
    missing = expected - set(palette)
    if missing:
        raise ValueError(
            f'Profile {name!r} is missing colors required by this node: '
            f'{sorted(missing)}'
        )
    for color_class in result:
        values = palette[color_class['name']]
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
