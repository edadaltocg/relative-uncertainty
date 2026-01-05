def set_nested(config: dict, key_path: str, value):
    """
    Given 'a.b.c', sets config['a']['b']['c'] = value (creating intermediate dicts).
    """
    keys = key_path.split('.')
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value