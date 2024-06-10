LAZY_IMPORT = True

if LAZY_IMPORT:
    import sys
    from ..utils.module_utils import _LazyModule
    _import_structure = {
        'waifu_tagger': ['WaifuTagger'],
        'waifu_scorer': ['WaifuScorer'],
    }
    sys.modules[__name__] = _LazyModule(__name__, globals()['__file__'], import_structure=_import_structure, module_spec=__spec__)

else:
    from .waifu_tagger import WaifuTagger
    from .waifu_scorer import WaifuScorer
