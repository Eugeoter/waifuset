import os


def open_file_folder(path: str):
    print(f"Open {path}")
    if path is None or path == "":
        return

    command = f'explorer /select,"{path}"'
    os.system(command)


EN2CN = {
    'category': '类别',
    'caption': '标注',
    'metadata': '元数据',
    'dataset': '数据集',
    'database': '数据库',
    'character': '角色',
    'main': '主菜单',
    'chunk': '页面',
    'log': '日志',
    'buffer': '缓冲区',
    'batch': '批处理',
    'showcase': '画廊',
    'quick tagging': '快速标注',
    'custom tagging': '自定义标注',
    'operational tagging': '高级标注',
    'optimizers': '优化标注',
    'wd14': 'WD14',
    'query': '查询',
    'regex': '正则表达式',
    'if': '如果',
    'include': '包含',
    'exclude': '排除',
    'add': '添加',
    'remove': '移除',
    'joiner': '连接符',
    'any': '任意',
    'all': '全部',
    'resolution': '分辨率',
    'image path': '图像路径',
    'info': '数据信息',
    'formalize': '格式化',
    'sort': '排序',
    'deduplicate': '去重',
    'de-overlap': '去重叠',
    'de-feature': '去特征',
    'general threshold': '阈值',
    'character threshold': '角色阈值',
    'os mode': '覆盖模式',
    'overwrite': '覆盖',
    'ignore': '忽略',
    'append': '追加',
    'prepend': '前置',
    'color': '色彩',
    'detail': '细致',
    'lowres': '低分辨率',
    'messy': '混乱',
    'aesthetic': '唯美',
    'beautiful': '美丽',
    'more': '更多',
    'image key': '数据指纹',
    'op': '操作',
    'tags': '标签',
    'inclusion': '包含关系',
    'threshold': '阈值',
    'character': '角色',
    'characters': '角色',
    'artist': '艺术家',
    'style': '风格',
    'styles': '风格',
    'quality': '质量',
    'and': '且',
    'or': '或',
}

CN2EN = {v: k for k, v in EN2CN.items()}


def en2cn(text):
    if text is None:
        return None
    return EN2CN.get(text.lower(), text)


def cn2en(text):
    if text is None:
        return None
    return CN2EN.get(text, text)


def translate(text, language='en'):
    translator = {
        'en': cn2en,
        'cn': en2cn,
    }
    translator = translator.get(language, cn2en)
    if isinstance(text, list):
        return [translator(t) for t in text]
    else:
        return translator(text)
