import re


def url_to_pid(url: str):
    match = re.search(r'(?:favorable|cheap)_(\d+)', url)
    return int(match.group(1))


def chinese_tags_to_bow(tags: list[str]) -> list[str]:

    def _gen():
        for tag in tags:
            for character in tag:
                yield character

    return list(_gen())
