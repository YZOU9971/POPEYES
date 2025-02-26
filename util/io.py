import json


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=True):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 4
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)


def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines
