import re
import json
import os
from ...const import ROOT


def search_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None


CUSTOM_TAG_PATH = search_file('custom_tags.json', ROOT)
TAG_TABLE_PATH = search_file('tag_table.json', ROOT)
PRIORITY_TABLE_PATH = search_file('priority_table.json', ROOT)
OVERLAP_TABLE_PATH = search_file('overlap_tags.json', ROOT)
FEATURE_TABLE_PATH = search_file('feature_table.json', ROOT)
ARTIST_TAGS_PATH = search_file('artist_tags.json', ROOT)
CHARACTER_TAGS_PATH = search_file('character_tags.json', ROOT)
COPYRIGHT_TAGS_PATH = search_file('copyright_tags.json', ROOT)

if not CUSTOM_TAG_PATH:
    print(f'custom tag config not found in root: {ROOT}')
if not PRIORITY_TABLE_PATH:
    print(f'priority table not found in root: {ROOT}')
if not TAG_TABLE_PATH:
    print(f'tag table not found in root: {ROOT}')
if not OVERLAP_TABLE_PATH:
    print(f'overlap table not found in root: {ROOT}')
if not FEATURE_TABLE_PATH:
    print(f'feature table not found in root: {ROOT}')

if not ARTIST_TAGS_PATH:
    print(f'artist tags not found in root: {ROOT}')
if not CHARACTER_TAGS_PATH:
    print(f'character tags not found in root: {ROOT}')
if not COPYRIGHT_TAGS_PATH:
    print(f'copyright tags not found in root: {ROOT}')

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

REGEX_CHARACTER = re.compile(PATTERN_CHARACTER)
REGEX_ARTIST = re.compile(PATTERN_ARTIST)
REGEX_STYLE = re.compile(PATTERN_STYLE)

CUSTOM_TAGS = None
QUALITY_TAGS = None
AESTHETIC_TAGS = None
STYLE_TAGS = None


def init_custom_tags(path=CUSTOM_TAG_PATH):
    global CUSTOM_TAGS, QUALITY_TAGS, AESTHETIC_TAGS, STYLE_TAGS
    if CUSTOM_TAGS is not None:
        return True
    try:
        with open(path, 'r') as f:
            custom_tag_table = json.load(f)
        custom_tag_table = {k: set(v) for k, v in custom_tag_table.items()}
        QUALITY_TAGS = custom_tag_table.get('quality', set())
        AESTHETIC_TAGS = custom_tag_table.get('aesthetic', set())
        STYLE_TAGS = custom_tag_table.get('style', set())
        CUSTOM_TAGS = QUALITY_TAGS | AESTHETIC_TAGS | STYLE_TAGS
        return True
    except Exception as e:
        CUSTOM_TAGS = None
        return False


def encode_tag(tag: str):
    tag = re.escape(tag)
    tag = REGEX_UNESCAPED_BRACKET.sub(r'\\\?\\\1', tag)
    tag = tag.replace('\ ', r'[\s_]')
    return tag


# PATTERN_CHARACTER_TAGS = '(' + '|'.join(
#     [encode_tag(tag) for tag in CHARACTER_TAGS]
# ) + ')'
# REGEX_CHARACTER_TAGS = re.compile(PATTERN_CHARACTER_TAGS)

TAG_TABLE = None
OVERLAP_TABLE = None
FEATURE_TABLE = None
PRIORITY_TABLE = None

ARTIST_TAGS = None
CHARACTER_TAGS = None
COPYRIGHT_TAGS = None


def init_artist_tags(path=ARTIST_TAGS_PATH):
    global ARTIST_TAGS
    if ARTIST_TAGS is not None:
        return True
    try:
        with open(path, 'r') as f:
            ARTIST_TAGS = set(json.load(f))
    except Exception as e:
        ARTIST_TAGS = None
        return False
    return True


def init_character_tags(path=CHARACTER_TAGS_PATH):
    global CHARACTER_TAGS
    if CHARACTER_TAGS is not None:
        return True
    try:
        with open(path, 'r') as f:
            CHARACTER_TAGS = set(json.load(f))
    except Exception as e:
        CHARACTER_TAGS = None
        return False
    return True


def init_copyright_tags(path=COPYRIGHT_TAGS_PATH):
    global COPYRIGHT_TAGS
    if COPYRIGHT_TAGS is not None:
        return True
    try:
        with open(path, 'r') as f:
            COPYRIGHT_TAGS = set(json.load(f))
    except Exception as e:
        COPYRIGHT_TAGS = None
        return False
    return True


def init_tag_table(table_path=TAG_TABLE_PATH):
    global TAG_TABLE
    if TAG_TABLE is not None:
        return True
    try:
        with open(table_path, 'r') as f:
            TAG_TABLE = json.load(f)
    except Exception as e:
        TAG_TABLE = None
        return False
    return True


def init_overlap_table(table_path=OVERLAP_TABLE_PATH):
    global OVERLAP_TABLE
    if OVERLAP_TABLE is not None:
        return True
    try:
        import json
        with open(table_path, 'r') as f:
            table = json.load(f)
        table = {entry['query']: (set(entry.get("has_overlap") or []), set(entry.get("overlap_tags") or [])) for entry in table}
        table = {k: v for k, v in table.items() if len(v[0]) > 0 or len(v[1]) > 0}
        OVERLAP_TABLE = table
        return True
    except Exception as e:
        OVERLAP_TABLE = None
        print(f'failed to read overlap table: {e}')
        return False


def init_priority_table(table_path=PRIORITY_TABLE_PATH):
    global PRIORITY_TABLE
    if PRIORITY_TABLE is not None:
        return True
    try:
        with open(table_path, 'r') as f:
            PRIORITY_TABLE = json.load(f)
    except Exception as e:
        PRIORITY_TABLE = None
        return False
    return True


def init_feature_table(table_path=FEATURE_TABLE_PATH, freq_thres=0.3, count_thres=1, least_sample_size=50):
    global FEATURE_TABLE
    if FEATURE_TABLE is not None:
        if not (freq_thres == FEATURE_TABLE.freq_thres and count_thres == FEATURE_TABLE.count_thres and least_sample_size == FEATURE_TABLE.least_sample_size):
            FEATURE_TABLE = None
            return init_feature_table(table_path, freq_thres, count_thres, least_sample_size)  # remake the table
        else:
            return True
    try:
        from .table import FeatureTable
        FEATURE_TABLE = FeatureTable(table_path, freq_thres=freq_thres, count_thres=count_thres, least_sample_size=least_sample_size)
    except Exception as e:
        FEATURE_TABLE = None
        return False
    return True


def init_priority_tags():
    global PRIORITY, PRIORITY_REGEX, PATTERN_CHARACTER_TAGS, REGEX_CHARACTER_TAGS
    if PRIORITY and PRIORITY_REGEX:
        return True

    if init_custom_tags():
        PATTERN_STYLE_TAGS = '(' + '|'.join([encode_tag(tag) for tag in STYLE_TAGS]) + ')'
    else:
        PATTERN_STYLE_TAGS = r''

    # ! spacing captions only.
    PRIORITY = {
        # Artist 0
        'artist': [PATTERN_ARTIST_TAG, PATTERN_ARTIST],
        # Role 1
        'role': [r'\d?\+?(?:boy|girl|other)s?', r'multiple (boys|girls|others)', 'no humans'],
        # Character 2
        # 'copyright': ['|'.join(GAME_TAGS)],
        'character': [PATTERN_CHARACTER, 'cosplay'],
        'race': [r'(furry|fox|pig|wolf|elf|oni|horse|cat|dog|arthropod|shark|mouse|lion|slime|tiger|raccoon|bird|squirrel|cow|animal|maid|sheep|bear|monster|mermaid|angel|demon|dark-skinned|mature|spider|fish|plant|goat|inkling|octoling) (female|male|girl|boy)s?',
                 'maid', 'nun', 'androgynous', 'demon', 'oni', 'giant', 'loli', 'angel', 'monster', 'office lady'],
        'solo': ['solo'],
        # Subject 6
        'subject': ['portrait', 'scenery', 'out-of-frame'],
        # Style 7
        'style': [PATTERN_STYLE_TAGS, PATTERN_STYLE],
        # Theme
        'theme': [r'.*\b(theme)\b.*', 'science fiction', 'fantasy'],
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

        # Artistic
        'aesthetic': ['|'.join(AESTHETIC_TAGS)],
        # Quality
        'quality': [r'\b(amazing|best|high|normal|low|worst|horrible) quality\b'],
    }

    PRIORITY_REGEX = [re.compile('|'.join([pattern for pattern in patterns if pattern.strip() != '']).replace(' ', r'[\s_]')) for patterns in PRIORITY.values()]

    return True


PRIORITY, PRIORITY_REGEX = None, None


PATTERN_CHARACTER_FEATURES = [
    r".*\b(hair|bang|braid|ahoge|eye|eyeshadow|eyelash|forehead|eyeliner|fang|eyebrow|pupil|tongue|makeup|lip|mole|ear|horn|nose|mole|tail|wing|breast|chest|tattoo|pussy|penis|fur|arm|leg|thigh|skin|freckle|leg|thigh|foot|feet|toe|finger)s?\b.*",
    r".*\b(twintails|ponytail|hairbun|double bun|hime cut|bob cut|sidelocks|loli|tan|eyelashes|halo)\b.*",
    r"\b(furry|fox|pig|wolf|elf|oni|horse|cat|dog|arthropod|shark|mouse|lion|slime|goblin|tiger|dragon|raccoon|bird|squirrel|cow|animal|maid|frog|sheep|bear|monster|mermaid|angel|demon|dark-skinned|mature|spider|fish|plant|goat|inkling|octoling|jiangshi)([\s_](girl|boy|other|male|female))?\b",
]
REGEX_CHARACTER_FEATURES = [re.compile(pattern.replace(' ', r'[\s_]')) for pattern in PATTERN_CHARACTER_FEATURES]


def get_key_index(key):
    if not PRIORITY:
        init_priority_tags()
    return list(PRIORITY.keys()).index(key)


LOWEST_PRIORITY = 999


def tag2priority(tag):
    if init_priority_table():  # query in table
        if tag.startswith("artist:"):
            return get_key_index('artist')
        elif tag.startswith("character:"):
            return get_key_index('character')
        elif tag.startswith("style:"):
            return get_key_index('style')
        elif 'quality' in tag:
            return get_key_index('quality')
        elif tag in AESTHETIC_TAGS:
            return get_key_index('aesthetic')
        else:
            tag = preprocess_tag(tag)
            if tag in PRIORITY_TABLE:
                return PRIORITY_TABLE[tag]
            else:
                return LOWEST_PRIORITY
    elif init_priority_tags():  # query in regex
        for i, regex in enumerate(PRIORITY_REGEX):
            if regex.match(tag):
                return i
        return LOWEST_PRIORITY
    else:
        return LOWEST_PRIORITY


def preprocess_tag(tag):
    tag = tag.strip().lower()
    tag = tag.replace('\\', '')
    tag = tag.replace(' ', '_')
    return tag


def sort_tags(tags):
    return sorted(tags, key=lambda x: tag2priority(x))


def query_tag_table(tag, default=None):
    if TAG_TABLE is None:
        init_tag_table()
    tag = tag.lower().replace('\\', '').replace(' ', '_').strip('_')
    return TAG_TABLE.get(tag, default)
