import re
import json
import numpy as np

OVERLAP_TABLE_PATH = './waifuset/json/overlap_tags.json'
CUSTOM_TAG_PATH = './waifuset/json/custom_tags.json'
WD_LABEL_PATH = './models/wd/selected_tags.csv'

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


def read_custom_tags(custom_tag_path):
    import json
    with open(custom_tag_path, 'r') as f:
        custom_tags = json.load(f)
    return custom_tags


try:
    CUSTOM_TAGS = read_custom_tags(CUSTOM_TAG_PATH)
except FileNotFoundError:
    CUSTOM_TAGS = {}
    print(f'custom tag config not found: {CUSTOM_TAG_PATH}')
except json.JSONDecodeError:
    CUSTOM_TAGS = {}
    print(f'custom tag config is invalid or broken: {CUSTOM_TAG_PATH}')
except Exception as e:
    CUSTOM_TAGS = {}
    print(f'failed to load custom tag config: {CUSTOM_TAG_PATH}, due to: {e}')


QUALITY_TAGS = set(CUSTOM_TAGS.get('quality', []))
AESTHETIC_TAGS = set(CUSTOM_TAGS.get('aesthetic', []))
STYLE_TAGS = set(CUSTOM_TAGS.get('style', []))
GAME_TAGS = set(CUSTOM_TAGS.get('game', []))
CHARACTER_TAGS = set(CUSTOM_TAGS.get('character', []))
CUSTOM_TAGS = QUALITY_TAGS | AESTHETIC_TAGS | STYLE_TAGS | GAME_TAGS

PATTERN_CHARACTER_TAGS = '(' + '|'.join([REGEX_UNESCAPED_BRACKET.sub(r'\\\?\\\1', tag).replace(' ', r'[\s_]') for tag in CHARACTER_TAGS]) + ')'
REGEX_CHARACTER_TAGS = re.compile(PATTERN_CHARACTER_TAGS)

PATTERN_STYLE_TAGS = '(' + '|'.join([REGEX_UNESCAPED_BRACKET.sub(r'\\\?\\\1', tag).replace(' ', r'[\s_]') for tag in STYLE_TAGS]) + ')'


def init_overlap_table():
    global OVERLAP_TABLE, OVERLAP_TABLE_PATH
    if OVERLAP_TABLE is not None:
        return
    import json
    with open(OVERLAP_TABLE_PATH, 'r') as f:
        table = json.load(f)
    table = {entry['query']: (set(entry.get("has_overlap") or []), set(entry.get("overlap_tags") or [])) for entry in table}
    table = {k: v for k, v in table.items() if len(v[0]) > 0 or len(v[1]) > 0}
    OVERLAP_TABLE = table


OVERLAP_TABLE = None  # ! need to be initialized by init_overlap_table() before use.


def init_wd14_tags():
    global WD_TAGS, WD_GENERAL_TAGS, WD_CHARACTER_TAGS, WD_LABEL_PATH
    if WD_TAGS and WD_GENERAL_TAGS and WD_CHARACTER_TAGS:
        return
    from pandas import read_csv
    df = read_csv(WD_LABEL_PATH)
    wd_tag_names = df["name"].tolist()
    wd_general_indexes = list(np.where(df["category"] == 0)[0])
    wd_general_tags = [wd_tag_names[i].replace('_', ' ') for i in wd_general_indexes]
    wd_character_indexes = list(np.where(df["category"] == 4)[0])
    wd_character_tags = [wd_tag_names[i].replace('_', ' ') for i in wd_character_indexes]
    wd_tags = wd_general_tags + wd_character_tags

    WD_TAGS = set(wd_tags)
    WD_GENERAL_TAGS = set(wd_general_tags)
    WD_CHARACTER_TAGS = set(wd_character_tags)


WD_TAGS, WD_GENERAL_TAGS, WD_CHARACTER_TAGS = None, None, None


def init_priority_tags():
    global PRIORITY, PRIORITY_REGEX, PATTERN_CHARACTER_TAGS, REGEX_CHARACTER_TAGS
    if PRIORITY and PRIORITY_REGEX:
        return

    # ! spacing captions only.
    PRIORITY = {
        # Artist
        'artist': [PATTERN_ARTIST_TAG, PATTERN_ARTIST],
        # Role
        'role': [r'\d?\+?(?:boy|girl|other)s?', r'multiple (boys|girls|others)', 'no humans'],
        # Character
        'copyright': ['|'.join(GAME_TAGS)],
        'character': [PATTERN_CHARACTER_TAGS, PATTERN_CHARACTER, 'cosplay'],
        'race': [r'(furry|fox|pig|wolf|elf|oni|horse|cat|dog|arthropod|shark|mouse|lion|slime|tiger|raccoon|bird|squirrel|cow|animal|maid|sheep|bear|monster|mermaid|angel|demon|dark-skinned|mature|spider|fish|plant|goat|inkling|octoling) (female|male|girl|boy)s?',
                 'maid', 'nun', 'androgynous', 'demon', 'oni', 'giant', 'loli', 'angel', 'monster', 'office lady'],
        'solo': ['solo'],
        # Subject
        'subject': ['portrait', 'scenery', 'out-of-frame'],
        # Style
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
        'skin': [r'[\w\-]+ skin', r'dark-skinned (?:female|male)', r'.*\b(tan|figure)\b.*'],

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
        'pussy': [r'.*\b(pussy|vaginal|penis)\b.*'],
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

    PRIORITY_REGEX = [re.compile('|'.join(patterns).replace(' ', r'[\s_]')) for patterns in PRIORITY.values()]


PRIORITY, PRIORITY_REGEX = None, None


PATTERN_CHARACTER_FEATURES = [
    r".*\b(hair|bang|braid|ahoge|eye|eyeshadow|eyeliner|eyebrow|pupil|tongue|lip|mole|ear|horn|tail|wing|breast|skin|freckle)s?\b.*",
    r".*\b(twintails|ponytail|hairbun|double bun|hime cut|bob cut|sidelocks|loli|tan|eyelashes|halo)\b.*",
    r"\b(furry|fox|pig|wolf|elf|oni|horse|cat|dog|arthropod|shark|mouse|lion|slime|tiger|raccoon|bird|squirrel|cow|animal|maid|sheep|bear|monster|mermaid|angel|demon|dark-skinned|mature|spider|fish|plant|goat|inkling|octoling)([\s_](girl|boy|other|male|female))?\b"
]
REGEX_CHARACTER_FEATURES = [re.compile(pattern.replace(' ', r'[\s_]')) for pattern in PATTERN_CHARACTER_FEATURES]
