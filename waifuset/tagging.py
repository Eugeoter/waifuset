import re
import json
import os
from huggingface_hub import hf_hub_download
from typing import Literal, List, Union, Dict
from . import logging

logger = logging.get_logger('tagging')


def search_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None


WIKI_REPO_ID = 'Eugeoter/waifuset-wiki'
WIKI_CACHE_DIR = None

WIKI_FILES = {
    'artist_tags': {
        'default': {
            'filename': 'artist_tags.txt',
        },
    },
    'character_tags': {
        'default': {
            'filename': 'character_tags.txt',
        },
    },
    'copyright_tags': {
        'default': {
            'filename': 'copyright_tags.txt',
        },
    },
    'meta_tags': {
        'default': {
            'filename': 'meta_tags.txt',
        },
    },
    'custom_tags': {
        'default': {
            'filename': 'custom_tags.json',
        },
    },
    'tag_priorities': {
        'default': {
            'filename': 'tag_priorities.json',
        },
    },
    'tag_implications': {
        'default': {
            'filename': 'tag_implications.json',
        },
    },
    'tag_aliases': {
        'default': {
            'filename': 'tag_aliases.json',
        },
    },
    'ch2physics': {
        'default': {
            'filename': 'ch2physics.json',
        },
    },
    'ch2clothes': {
        'default': {
            'filename': 'ch2clothes.json',
        },
    },
    'ch2sex': {
        'default': {
            'filename': 'ch2sex.json',
        },
    },
}

ALL_TAG_TYPES = ('artist', 'character', 'style', 'quality', 'aesthetic', 'copyright', 'meta', 'safety', 'year', 'period')

PATTERN_ARTIST_TAG = r"(?:^|,\s)(by[\s_]([\w\d][\w_\-.\s()\\]*))"  # match `by xxx`
PATTERN_QUALITY_TAG = r'\b((amazing|best|high|normal|low|worst|horrible)([\s_]quality))\b'  # match `xxx quality`
PATTERN_UNESCAPED_BRACKET = r"(?<!\\)([\(\)\[\]\{\}])"  # match `(` and `)`
PATTERN_ESCAPED_BRACKET = r"\\([\(\)\[\]\{\}])"  # match `\(` and `\)`
PATTERN_WEIGHTED_CAPTION = r"[^\\]\((.+?)(?::([\d\.]+))?[^\\]\)"  # match `(xxx:yyy)`

REGEX_ARTIST_TAG = re.compile(PATTERN_ARTIST_TAG)
REGEX_UNESCAPED_BRACKET = re.compile(PATTERN_UNESCAPED_BRACKET)
REGEX_ESCAPED_BRACKET = re.compile(PATTERN_ESCAPED_BRACKET)
REGEX_WEIGHTED_CAPTION = re.compile(PATTERN_WEIGHTED_CAPTION)
REGEX_QUALITY_TAG = re.compile(PATTERN_QUALITY_TAG)

PATTERN_CHARACTER = r"((character:[\s_]*)([^,]+))"
PATTERN_ARTIST = r"((artist:[\s_]*)([^,]+))"
PATTERN_STYLE = r"((style:[\s_]*)([^,]+))"
PATTERN_COPYRIGHT = r"((copyright:[\s_]*)([^,]+))"
PATTERN_META = r"((meta:[\s_]*)([^,]+))"
PATTERN_QUALITY = r"((quality:[\s_]*)([^,]+))"
PATTERN_SAFETY = r"((safety:[\s_]*)([^,]+))"
PATTERN_YEAR = r"((year:[\s_]*)([^,]+))"
PATTERN_PERIOD = r"((period:[\s_]*)([^,]+))"


REGEX_CHARACTER = re.compile(PATTERN_CHARACTER)
REGEX_ARTIST = re.compile(PATTERN_ARTIST)
REGEX_STYLE = re.compile(PATTERN_STYLE)
REGEX_COPYRIGHT = re.compile(PATTERN_COPYRIGHT)
REGEX_META = re.compile(PATTERN_META)
REGEX_QUALITY = re.compile(PATTERN_QUALITY)
REGEX_SAFETY = re.compile(PATTERN_SAFETY)
REGEX_YEAR = re.compile(PATTERN_YEAR)
REGEX_PERIOD = re.compile(PATTERN_PERIOD)


CUSTOM_TAGS = None
QUALITY_TAGS = None
AESTHETIC_TAGS = None
STYLE_TAGS = None

TAG_IMPLICATIONS = None
TAG_ALIASES = None
TAG_PRIORITIES = None
CH2PHYSICS = None
CH2CLOTHES = None
CH2SEX = None

ARTIST_TAGS = None
CHARACTER_TAGS = None
COPYRIGHT_TAGS = None
META_TAGS = None

SAFE_LEVEL_TO_SAFETY_TAG = {
    'g': 'general',
    's': 'sensitive',
    'q': 'questionable',
    'e': 'explicit',
}
DEFAULT_FEATURE_TYPE_TO_FREQUENCY_THRESHOLD = {
    'physics': 0.25,
    'clothes': 0.35,
    'sex': 0.55,
}
LOWEST_TAG_PRIORITY = 999

TAG_PRIORITY_PATTERN, TAG_PRIORITY_REGEX = None, None
CHARACTER_PHYSICS_FEATURE_PATTERN, CHARACTER_PHYSICS_FEATURE_REGEX = None, None

# ======================================== wiki cache ========================================


def set_wiki_cache_dir(cache_dir):
    global WIKI_CACHE_DIR
    WIKI_CACHE_DIR = cache_dir


# ======================================== tag transforms ========================================


def fmt2unescape(tag):
    return re.sub(r'(\\+)([\(\)])', r'\2', tag)


def fmt2escape(tag):
    return re.sub(r'(?<!\\)(\()(.*)(?<!\\)(\))', r'\\\1\2\\\3', tag)  # negative lookbehind


def fmt2danbooru(tag):
    tag = tag.lower().replace(' ', '_').strip('_')
    tag = re.sub(r'(_+)', '_', tag)
    tag = tag.replace(':_', ':')
    tag = fmt2unescape(tag)
    return tag


def fmt2train(tag):
    tag = fmt2danbooru(tag)
    tag = tag.replace('_', ' ')
    return tag


def fmt2prompt(tag):
    tag = tag.replace('_', ' ').strip()
    tag = fmt2escape(tag)
    tag = tag.replace(': ', ':')
    return tag


def fmt2awa(tag):
    tag = fmt2prompt(tag)
    if (tagtype := get_tagtype_from_tag(tag)):
        prefix, tag = tag.split(":", 1)
        if prefix == 'artist':
            return f"by {tag}"
        elif prefix == 'character':
            return f"1 {tag}"
        elif prefix == 'style':
            return f"{tag} style"
        elif prefix == 'quality':
            return f"{tag} quality" if not tag.endswith(' quality') else tag
        elif prefix == 'safety':
            return tag
    return tag


def match(pattern, tag):
    if isinstance(pattern, str):
        return tag == pattern
    elif isinstance(pattern, re.Pattern):
        return re.match(pattern, tag)


def uncomment_tag(tag: str, tagtype=None):
    tagtype = tagtype or get_tagtype_from_tag(tag)
    return tag[len(tagtype) + 1:] if tagtype else tag


def comment_tag(tag: str, tagtype=None):
    return f'{tagtype}:{tag}' if tagtype else tag

# ======================================== tagtype ========================================


def get_tags_from_tagtype(tagtype: Literal['artist', 'character', 'style', 'aesthetic', 'copyright', 'quality', 'meta', 'safety', 'year', 'period']) -> Union[set, None]:
    r"""
    Get specific tagset.
    """
    if tagtype == 'artist':
        return get_artist_tags()
    elif tagtype == 'character':
        return get_character_tags()
    elif tagtype == 'style':
        return get_style_tags()
    elif tagtype == 'aesthetic':
        return get_aesthetic_tags()
    elif tagtype == 'copyright':
        return get_copyright_tags()
    elif tagtype == 'quality':
        return get_quality_tags()
    elif tagtype == 'meta':
        return get_meta_tags()
    else:
        raise ValueError(f'invalid tagtype: {tagtype}')


def get_tagtype_from_tag(tag: str):
    if ':' in tag:
        for tagtype in ALL_TAG_TYPES:
            if tag.startswith(tagtype + ':'):
                return tagtype
    return None


def get_tagtype_from_wiki(tag: str):
    dan_tag = fmt2danbooru(tag)
    if dan_tag in get_artist_tags():
        return 'artist'
    elif dan_tag in get_character_tags():
        return 'character'
    elif dan_tag in get_style_tags():
        return 'style'
    elif dan_tag in get_copyright_tags():
        return 'copyright'
    elif dan_tag in get_meta_tags():
        return 'meta'
    elif dan_tag in get_quality_tags():
        return 'quality'
    else:
        return None


def get_tagtype(tag: str):
    return get_tagtype_from_tag(tag) or get_tagtype_from_wiki(tag)

# ======================================== tagset functions ========================================


def get_artist_tags(wiki_name_or_path=None):
    global ARTIST_TAGS
    if ARTIST_TAGS is not None:
        return ARTIST_TAGS
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['artist_tags']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            ARTIST_TAGS = set(f.read().splitlines())
    except Exception as e:
        ARTIST_TAGS = None
        logger.error(f'failed to read artist tags: {e}')
        return None
    return ARTIST_TAGS


def get_character_tags(wiki_name_or_path=None):
    global CHARACTER_TAGS
    if CHARACTER_TAGS is not None:
        return CHARACTER_TAGS
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['character_tags']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            CHARACTER_TAGS = set(f.read().splitlines())
    except Exception as e:
        CHARACTER_TAGS = None
        logger.error(f'failed to read character tags: {e}')
        return None
    return CHARACTER_TAGS


def get_copyright_tags(wiki_name_or_path=None):
    global COPYRIGHT_TAGS
    if COPYRIGHT_TAGS is not None:
        return COPYRIGHT_TAGS
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['copyright_tags']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            COPYRIGHT_TAGS = set(f.read().splitlines())
    except Exception as e:
        COPYRIGHT_TAGS = None
        logger.error(f'failed to read copyright tags: {e}')
        return None
    return COPYRIGHT_TAGS


def get_meta_tags(wiki_name_or_path=None):
    global META_TAGS
    if META_TAGS is not None:
        return META_TAGS
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['meta_tags']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            META_TAGS = set(f.read().splitlines())
    except Exception as e:
        META_TAGS = None
        logger.error(f'failed to read meta tags: {e}')
        return None
    return META_TAGS

# ======================================== custom tag functions ========================================


def get_custom_tags(wiki_name_or_path=None):
    global CUSTOM_TAGS, QUALITY_TAGS, AESTHETIC_TAGS, STYLE_TAGS
    if CUSTOM_TAGS is not None:
        return CUSTOM_TAGS
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['custom_tags']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            custom_tag_table = json.load(f)
        custom_tag_table = {k: set(v) for k, v in custom_tag_table.items()}
        QUALITY_TAGS = custom_tag_table.get('quality', set())
        AESTHETIC_TAGS = custom_tag_table.get('aesthetic', set())
        STYLE_TAGS = custom_tag_table.get('style', set())
        CUSTOM_TAGS = QUALITY_TAGS | AESTHETIC_TAGS | STYLE_TAGS
        return CUSTOM_TAGS
    except Exception as e:
        CUSTOM_TAGS = None
        logger.error(f'failed to read custom tags: {e}')
        return None


def get_aesthetic_tags():
    get_custom_tags()
    return AESTHETIC_TAGS


def get_style_tags():
    get_custom_tags()
    return STYLE_TAGS


def get_quality_tags():
    get_custom_tags()
    return QUALITY_TAGS

# ======================================== tag implication, alias, priority functions ========================================


def get_tag_implications(wiki_name_or_path=None):
    global TAG_IMPLICATIONS
    if TAG_IMPLICATIONS is not None:
        return TAG_IMPLICATIONS
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['tag_implications']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            TAG_IMPLICATIONS = json.load(f)
    except Exception as e:
        TAG_IMPLICATIONS = None
        logging.error(f'failed to read implication table: {e}')
        return None
    return TAG_IMPLICATIONS


def get_tag_aliases(wiki_name_or_path=None):
    global TAG_ALIASES
    if TAG_ALIASES is not None:
        return TAG_ALIASES
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['tag_aliases']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            TAG_ALIASES = json.load(f)
    except Exception as e:
        TAG_ALIASES = None
        logging.error(f'failed to read alias table: {e}')
        return None
    return TAG_ALIASES


def get_tag_priorities(wiki_name_or_path=None):
    global TAG_PRIORITIES
    if TAG_PRIORITIES is not None:
        return TAG_PRIORITIES
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['tag_priorities']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            TAG_PRIORITIES = json.load(f)
    except Exception as e:
        TAG_PRIORITIES = None
        logging.info(f'failed to read tag priorities: {e}')
        return None
    return TAG_PRIORITIES

# ======================================== ch2feature functions ========================================


def get_ch2physics(wiki_name_or_path=None):
    global CH2PHYSICS
    if CH2PHYSICS is not None:
        return CH2PHYSICS
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['ch2physics']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            CH2PHYSICS = json.load(f)
    except Exception as e:
        CH2PHYSICS = None
        logging.error(f'failed to read ch2physics: {e}')
        return None
    return CH2PHYSICS


def get_ch2clothes(wiki_name_or_path=None):
    global CH2CLOTHES
    if CH2CLOTHES is not None:
        return CH2CLOTHES
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['ch2clothes']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            CH2CLOTHES = json.load(f)
    except Exception as e:
        CH2CLOTHES = None
        logging.error(f'failed to read ch2clothes: {e}')
        return None
    return CH2CLOTHES


def get_ch2sex(wiki_name_or_path=None):
    global CH2SEX
    if CH2SEX is not None:
        return CH2SEX
    try:
        wiki_name_or_path = wiki_name_or_path or hf_hub_download(WIKI_REPO_ID, filename=WIKI_FILES['ch2sex']['default']['filename'], repo_type='dataset', cache_dir=WIKI_CACHE_DIR)
        with open(wiki_name_or_path, 'r', encoding='utf-8') as f:
            CH2SEX = json.load(f)
    except Exception as e:
        CH2SEX = None
        logging.error(f'failed to read ch2sex: {e}')
        return None
    return CH2SEX


def get_ch2feature2ratio(character: str, feature_types: List[Literal['physics', 'clothes', 'sex']] = ['physics', 'clothes', 'sex']) -> Dict[str, float]:
    if isinstance(feature_types, str):
        feature_types = [feature_types]
    character = uncomment_tag(character)
    character = fmt2danbooru(character)
    features = {}
    if 'physics' in feature_types:
        ch2physics = get_ch2physics()
        features.update(ch2physics.get(character, {}))
    if 'clothes' in feature_types:
        ch2clothes = get_ch2clothes()
        features.update(ch2clothes.get(character, {}))
    if 'sex' in feature_types:
        ch2sex = get_ch2sex()
        features.update(ch2sex.get(character, {}))
    return features


def get_character_features(character: str, feature_type_to_frequency_threshold: Dict[Literal['physics', 'clothes', 'sex'], float] = DEFAULT_FEATURE_TYPE_TO_FREQUENCY_THRESHOLD) -> List[str]:
    character = uncomment_tag(character)
    character = fmt2danbooru(character)
    features = []
    if 'physics' in feature_type_to_frequency_threshold:
        ch2physics = get_ch2physics()
        threshold = feature_type_to_frequency_threshold['physics']
        features.extend([feature for feature, ratio in ch2physics.get(character, {}).items() if ratio >= threshold])
    if 'clothes' in feature_type_to_frequency_threshold:
        ch2clothes = get_ch2clothes()
        threshold = feature_type_to_frequency_threshold['clothes']
        features.extend([feature for feature, ratio in ch2clothes.get(character, {}).items() if ratio >= threshold])
    if 'sex' in feature_type_to_frequency_threshold:
        ch2sex = get_ch2sex()
        threshold = feature_type_to_frequency_threshold['sex']
        features.extend([feature for feature, ratio in ch2sex.get(character, {}).items() if ratio >= threshold])
    return features

# ======================================== tag priority functions ========================================


def get_tag_priorities_regex():
    global TAG_PRIORITY_PATTERN, TAG_PRIORITY_REGEX
    if TAG_PRIORITY_PATTERN and TAG_PRIORITY_REGEX:
        return TAG_PRIORITY_REGEX

    def compile_or_regex(tags):
        return '(' + '|'.join(tags) + ')' if tags else ''

    if get_custom_tags():
        PATTERN_STYLE_TAGS = compile_or_regex(STYLE_TAGS)
    else:
        PATTERN_STYLE_TAGS = r''

    try:
        # ! spacing captions only.
        TAG_PRIORITY_PATTERN = {
            # Role
            'role': [r'\d?\+?(?:boy|girl|other)s?', r'multiple (boys|girls|others)', 'no humans'],
            # Character
            'character': [
                PATTERN_CHARACTER,
                'cosplay'
            ],
            # Copyright
            'copyright': [
                PATTERN_COPYRIGHT,
                # compile_or_regex(COPYRIGHT_TAGS),
            ],
            'race': [r'(furry|fox|pig|wolf|elf|oni|horse|cat|dog|arthropod|shark|mouse|lion|slime|tiger|raccoon|bird|squirrel|cow|animal|maid|sheep|bear|monster|mermaid|angel|demon|dark-skinned|mature|spider|fish|plant|goat|inkling|octoling) (female|male|girl|boy)s?',
                     'maid', 'nun', 'androgynous', 'demon', 'oni', 'giant', 'loli', 'angel', 'monster', 'office lady'],
            'solo': ['solo'],
            # Subject
            'subject': ['portrait', 'scenery', 'out-of-frame'],
            # Theme
            'theme': [r'.*\b(theme)\b.*', 'science fiction', 'fantasy'],
            # Safety
            'safety': [r'\b(safety:\s*)?(general|sensitive|questionable|explicit)\b'],
            # Environment
            'environment': ['nature', 'indoors', 'outdoors'],
            # Background
            'background': [r'.*\bbackground\b.*'],
            # Angle
            'angle': [r'from (side|behind|above|below)', r'(full|upper|lower) body', r'.*\b(focus)\b.*', 'cowboy shot', 'close-up', 'dutch angle', 'wide shot', 'multiple views', r'.*\b(out of frame)\b.*', 'selfie'],

            # Actions
            'action': [r'.*\b(sitting|lying|soaked|outstretched|standing|masturbation|kneeling|crouching|squatting|stretching|bespectacled|leaning|looking|kissing|sex|sewing|facing|carrying|licking|floating|wading|aiming|reaching|drinking|drawing|fidgeting|covering|tying|walking|running|jumping|protecting|fighting|inkling|grabing|eating|trembling|sleeping|crying|straddling|pointing|drooling)\b.*',
                       'flying', 'falling', 'diving', 'holding', "jack-o' challenge", r'(hand|arm|keg|thigh)s? (up|down)', 'heart hands', 'cowgirl position', 'lifted by self', 'hetero', 'paw pose'],
            'additional_action': ['on back', 'on stomach'],

            # Expressions
            'expression': [r'.*\b(happy|sad|angry|grin|surprised|scared|embarrassed|shy|smiling|smile|frowning|crying|laughing|blushing|sweating|blush|:3|:o|expression|expressionless)\b.*'],

            # Skin
            'skin': [r'dark-skinned (?:female|male)', r'.*\b(tan|figure|skin)\b.*'],

            # Features
            'face_feature': [r'.*\b(ear|horn|tail|mouth|lip|teeth|tongue|fang|saliva|kemonomimi mode|mustache|beard|sweatdrop)s?\b.*'],

            # Eyes
            'eye': [r'.*\beyes\b.*', 'heterochromia'],
            'eye_feature': [r'.*\b(eyelashes|eyeshadow|eyebrow|eye|pupil)s?\b.*'],
            'eye_accessory': [r'.*\b(eyepatch|glasses|sunglassess|eyewear|goggles|makeup)\b.*'],

            # Hair
            'hair': [r'[\w\-\s]+ hair'],
            'hairstyle': [r'.*\b(hair|ponytail|twintail|hairbun|bun|bob cut|braid|bang|ahoge)s?\b.*', 'ringlets', 'sidelocks', 'fringe', 'forelock', 'two side up'],
            # Hair ornaments
            'hair_ornament': [r'.*\b(hairclip|haircut|hairband|hair ornament)s?\b.*'],

            'figure': ['plump',],

            # Breast
            'breast': [r'.*\b(huge|large|medium|small|flat) (breasts|chest)\b'],
            'breast_feature': [r'.*\b(breast|chest)s?\b.*', r'(side|inner|under)boob', 'cleavage'],
            'nipple': [r'.*\b(nipple|areola|areolae)s?\b.*'],

            # Pussy
            'pussy': [r'.*\b(pussy|vaginal|penis|anus)\b.*'],
            'mosaic': [r'.*\b(uncensor|censor)(ed|ing)?\b.*'],

            # Bodies
            'body': [r'.*\b(ass|butt|booty|rear|navel|groin|armpit|hip|thigh|leg|feet|foot)s?\b.*', 'barefoot'],
            'body_feature': [r'.*\b(mole|tattoo|scar|bandaid|bandage|blood|sweat|tear)s?\b.*', 'freckles', 'body freckles', 'collarbone', 'navel', 'belly button', 'piercing', 'birthmark', 'wound', 'bruise'],

            # Suit
            'suit': [r'.*\b(enmaided|plugsuit|nude)\b.*'],
            # Clothing
            'clothes': [r'.*\b(clothes|outfit|suit|capelet|headwear|maid|apron|vest|cloak|kneehighs|petticoat|legwear|serafuku|dress|sweater|hoodie|uniform|armor|veil|footwear|thighhigh|clothing|garment|attire|robe|kimono|shirt|skirt|pants|shorts|shoes|boots|gloves|socks|stockings|pantyhose|bra|panties|underwear|lingerie|swimsuit|bikini|bodysuit|leotard|tights|coat|jacket|cape|scarf|hat|cap|glasses|sunglasses|mask|helmet|headphones)s?\b.*',
                        'bottomless', 'topless', 'official alternate costume', 'alternate costume', r'.*\bnaked.*\b'],
            # Clothing Features
            'clothes_accessory': [r'.*\b(center opening|pelvic curtain|high heels|choker|zettai ryouiki|tassel|bow|sleeve|necktie|neckline|skindentation|highleg|gown|halterneck|turtleneck|collar|bowtie|fishnets|cutout|ribbon|sleeveless|crossdressing|hood|shoulder|belt|frills|halo|jewelry)s?\b.*'],

            # Fingers
            'digit': [r'.*\b(digit|finger|toe)s?\b.*', 'v', r'.*\b(gesture)\b.*'],
            'nail': [r'.*\b(fingernail|toenail|nail)s?\b.*'],

            # Items
            'item': [r'.*\b(weapon|tool|katana|instrument|gadget|device|equipment|item|object|artifact|accessory|prop|earrings|necklace|bracelet|ring|watch|bag|backpack|purse|umbrella|parasol|cane|spear|sword|knife|gun|pistol|revolver|shotgun|rifle|gun|cannon|rocket launcher|grenade|bomb|shield|wing|hoove|antler)s?\b.*'],

            # Artist
            'artist': [
                PATTERN_ARTIST_TAG,
                PATTERN_ARTIST,
            ],
            # Style
            'style': [
                PATTERN_STYLE_TAGS,
                PATTERN_STYLE,
            ],
            # Artistic
            # 'aesthetic': [compile_or_regex(AESTHETIC_TAGS)],
            # Quality
            'quality': [
                r'\b(amazing|best|high|normal|low|worst|horrible) quality\b',
                'masterpiece',
                r'quality:.*',
            ],
            # Meta
            'meta': [
                PATTERN_META,
                # compile_or_regex(META_TAGS),
            ],
        }

        TAG_PRIORITY_REGEX = [re.compile('|'.join([pattern for pattern in patterns if pattern.strip() != '']).replace(' ', r'[\s_]')) for patterns in TAG_PRIORITY_PATTERN.values()]

    except Exception as e:
        TAG_PRIORITY_PATTERN, TAG_PRIORITY_REGEX = None, None
        logging.error(f'failed to compile tag priorities: {e}')
        return None

    return TAG_PRIORITY_REGEX


def get_character_physics_feature_regex():
    global CHARACTER_PHYSICS_FEATURE_PATTERN, CHARACTER_PHYSICS_FEATURE_REGEX
    if CHARACTER_PHYSICS_FEATURE_PATTERN and CHARACTER_PHYSICS_FEATURE_REGEX:
        return True

    CHARACTER_PHYSICS_FEATURE_PATTERN = [
        r".*\b(hair|bang|braid|ahoge|eye|eyeshadow|eyelash|forehead|eyeliner|fang|eyebrow|pupil|tongue|makeup|lip|mole|ear|horn|nose|mole|tail|wing|breast|chest|tattoo|pussy|penis|fur|arm|leg|thigh|skin|freckle|leg|thigh|foot|feet|toe|finger)s?\b.*",
        r".*\b(twintails|ponytail|hairbun|double bun|hime cut|bob cut|sidelocks|loli|tan|eyelashes|halo)\b.*",
        r"\b(furry|fox|pig|wolf|elf|oni|horse|cat|dog|arthropod|shark|mouse|lion|slime|goblin|tiger|dragon|raccoon|bird|squirrel|cow|animal|maid|frog|sheep|bear|monster|mermaid|angel|demon|dark-skinned|mature|spider|fish|plant|goat|inkling|octoling|jiangshi)([\s_](girl|boy|other|male|female))?\b",
    ]
    CHARACTER_PHYSICS_FEATURE_REGEX = [re.compile(pattern.replace(' ', r'[\s_]')) for pattern in CHARACTER_PHYSICS_FEATURE_PATTERN]

    return True


def get_tag_priority_from_tag_category(tag_category):
    if not TAG_PRIORITY_PATTERN:
        get_tag_priorities_regex()
    return list(TAG_PRIORITY_PATTERN.keys()).index(tag_category) if tag_category in TAG_PRIORITY_PATTERN else LOWEST_TAG_PRIORITY


def get_tag_priority(tag):
    r"""
    Convert a tag to a priority. Lower priority means higher importance.
    """
    # priority from tag type
    if ':' in tag and any(tag.startswith(tagtype + ':') for tagtype in ALL_TAG_TYPES):
        return get_tag_priority_from_tag_category(tag.split(':', 1)[0])
    # priority from quality
    elif tag.endswith('quality'):
        return get_tag_priority_from_tag_category('quality')
    # priority from aesthetic
    elif get_custom_tags() and tag in AESTHETIC_TAGS:
        return get_tag_priority_from_tag_category('aesthetic')
    # priority from tag priorities table
    elif get_tag_priorities():
        dan_tag = fmt2danbooru(tag)
        if dan_tag in TAG_PRIORITIES:
            return TAG_PRIORITIES[dan_tag]
        else:
            return LOWEST_TAG_PRIORITY
    # priority from regex matching
    elif get_tag_priorities_regex():
        for i, regex in enumerate(TAG_PRIORITY_REGEX):
            if regex.match(tag):
                return i
        return LOWEST_TAG_PRIORITY
    # otherwise, lowest priority
    else:
        return LOWEST_TAG_PRIORITY
