# A simple interface to utilize Hakubooru (original github: https://github.com/KohakuBlueleaf/HakuBooru)

import os
import sqlite3

TAG_ID_2_TYPE = {
    0: "general",
    1: "artist",
    2: "character",
    3: "copyright",
    4: "meta",
}

RATING_ID_2_TYPE = {
    0: "g",
    1: "s",
    2: "q",
    3: "e",
}


class Hakubooru:
    def __init__(self, db_file):
        if not os.path.exists(db_file):
            raise FileNotFoundError(f"database file {db_file} not found")
        self.conn = sqlite3.connect(db_file)
        self.header = get_header(self.conn, 'post')
        self.cursor = self.conn.cursor()  # cursor to the post table
        self._tag2id = None
        self._tagbase = None
        self._tagwiki = None

    @property
    def tagbase(self):
        r"""
        Tag database. A dict with tag_id as key and tag metadata as value.
        """
        if self._tagbase is None:
            self._tagbase = get_tagbase(self.conn)
        return self._tagbase

    @property
    def tagwiki(self):
        if self._tagwiki is None:
            self._tagwiki = get_tagwiki(self.conn)
        return self._tagwiki

    @property
    def tag2id(self):
        r"""
        Dict to map tag name to tag id.
        """
        if self._tag2id is None:
            db = self._tagbase or self._tagwiki
            if not db:
                db = self.tagbase
            self._tag2id = get_tag2id(db)
        return self._tag2id

    def __iter__(self):
        self.cursor.execute("SELECT * FROM post")
        return self

    def __next__(self):
        r"""
        Iterate through the post table and return metadata in danbooru format of each post.
        """
        row = self.cursor.fetchone()
        if row is None:
            raise StopIteration
        datadict = data2dict(row, self.header)
        metadata = parse_datadict(datadict, tagbase=self.tagbase)
        return metadata

    def __len__(self):
        self.cursor.execute("SELECT COUNT(*) FROM post")
        return self.cursor.fetchone()[0]

    def get_tag_wiki(self, tag):
        tag_id = self.tag2id[tag]
        return self.tagwiki.get(tag_id)

    def query(
        self,
        tags,
    ):
        tag_ids = [dec_to_base36(self.tag2id[tag]) for tag in tags]
        tag_ids = set(tag_ids)
        cond = ' AND '.join(f'tag_list LIKE \'%#{tag_id}#%\'' for tag_id in tag_ids)
        self.cursor.execute(f"SELECT * FROM post WHERE {cond}")
        rows = self.cursor.fetchall()
        print(f"find {len(rows)} posts")
        for row in rows:
            datadict = data2dict(row, self.header)
            metadata = parse_datadict(datadict, tagbase=self.tagbase)
            yield metadata


def dec_to_base36(num):
    if num < 0:
        return '-' + dec_to_base36(-num)

    # 定义36进制中使用的字符
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"

    # 处理0的特殊情况
    if num == 0:
        return "0"

    result = ""
    while num > 0:
        # 计算余数并添加到结果字符串
        result = digits[num % 36] + result
        num //= 36

    return result


def get_tagbase(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tag")
    rows = cursor.fetchall()
    db = {}
    for row in rows:
        id, name, type_id, count = row
        db[id] = {"id": id, "name": name, "count": count, "type": TAG_ID_2_TYPE[type_id]}
    cursor.close()
    return db


def get_tagwiki(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tag_wikis")
    rows = cursor.fetchall()
    db = {}
    for row in rows:
        id, name, desc = row[:3]
        db[id] = {"id": id, "name": name, "desc": desc}
    cursor.close()
    return db


def get_tag2id(db):
    return {tag['name']: tag['id'] for tag in db.values()}


def data2dict(row, header):
    assert len(row) == len(header)
    return {k: v for k, v in zip(header, row)}


def parse_datadict(dic, tagbase):
    dic['rating'] = RATING_ID_2_TYPE[dic['rating']]
    dic['has_children'] = bool(dic['has_children'])
    dic['has_active_children'] = bool(dic['has_active_children'])
    dic['has_large'] = bool(dic['has_large'])
    dic['has_visible_children'] = bool(dic['has_visible_children'])
    tag_metalist = [tagbase.get(int(tag36, 36), None) for tag36 in dic.pop('tag_list')[1:-1].split('#$')]
    tag_metalist = [tag for tag in tag_metalist if tag is not None]
    tag_list = [tag_md['name'] for tag_md in tag_metalist]
    tag_list_general = []
    tag_list_artist = []
    tag_list_character = []
    tag_list_copyright = []
    tag_list_meta = []
    for tag_md in tag_metalist:
        if tag_md['type'] == 'general':
            tag_list_general.append(tag_md['name'])
        elif tag_md['type'] == 'artist':
            tag_list_artist.append(tag_md['name'])
        elif tag_md['type'] == 'character':
            tag_list_character.append(tag_md['name'])
        elif tag_md['type'] == 'copyright':
            tag_list_copyright.append(tag_md['name'])
        elif tag_md['type'] == 'meta':
            tag_list_meta.append(tag_md['name'])
    dic['tags'] = tag_list
    dic['tags_general'] = tag_list_general
    dic['tags_artist'] = tag_list_artist
    dic['tags_character'] = tag_list_character
    dic['tags_copyright'] = tag_list_copyright
    dic['tags_meta'] = tag_list_meta

    return dic


def get_header(conn, table):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table});")
    columns_info = cursor.fetchall()
    cols = []
    for col in columns_info:
        cols.append(col[1])
    return tuple(cols)
