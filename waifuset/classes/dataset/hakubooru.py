# A simple interface to utilize Hakubooru (original github: https://github.com/KohakuBlueleaf/HakuBooru)

import os
import requests
import json
from huggingface_hub import hf_hub_download
from .sqlite3_dataset import SQLite3Dataset

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

REPO_ID = "KBlueLeaf/danbooru2023-metadata-database"
DB_FILE = "danbooru2023.db"


class Hakubooru(SQLite3Dataset):
    def __init__(self, fp, **kwargs):
        super().__init__(fp, **kwargs)
        self.set_table('post')
        self._tag2id = None
        self._tagbase = None
        self._tagwiki = None

    @classmethod
    def from_huggingface(cls, repo_id=None, filename=None, cache_dir=None):
        repo_id = repo_id or REPO_ID
        filename = filename or DB_FILE
        db_file = hf_hub_download(
            repo_id,
            filename,
            cache_dir=cache_dir,
        )
        return cls(db_file)

    def query_post(
        self,
        tags,
        postprocess=True,
    ):
        r"""
        Query posts that contains ALL the tags.
        """
        tags = tags if isinstance(tags, list) else [tags]
        tag_ids = [dec_to_base36(self.tag2id[tag]) for tag in tags]
        cond = ' AND '.join(f"tag_list GLOB \'*[$#]{tag_id}#*\'" for tag_id in tag_ids)
        self.cursor.execute(f"SELECT * FROM post WHERE {cond}")
        rows = self.cursor.fetchall()
        for row in rows:
            yield self.postprocessor(row, enable=postprocess)

    def query_tag_info(self, tag):
        r"""
        Query tag metadata by tag name.
        """
        tag_id = self.tag2id.get(tag)
        if tag_id is None:
            return None
        return self.tagbase.get(tag_id)

    def query_tag_wiki(self, tag):
        r"""
        Query tag wiki by tag name.
        """
        tag_id = self.tag2id.get(tag)
        if tag_id is None:
            return None
        return self.tagwiki.get(tag_id)

    def postprocessor(self, row, enable=True):
        r"""
        Postprocess a row to a dict.
        """
        if not enable:
            return row
        datadict = super().postprocessor(row, enable)
        return parse_datadict(datadict, tagbase=self.tagbase)

    @property
    def tagbase(self):
        r"""
        Tag database. A dict with tag id as key and tag metadata as value.
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

    def download_one(
        self,
        info,
        directory,
        save_image=True,
        save_metadata=True,
    ):
        url = info['file_url']
        id = info['id']
        stem = f"danbooru_{id}"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if save_image:
            ext = os.path.splitext(url)[1]
            img_path = os.path.join(directory, f"{stem}{ext}")
            try:
                download(url, img_path)
            except Exception as e:
                self.logger.print(f"Failed to download {url}")
        if save_metadata:
            meta_path = os.path.join(directory, f"{stem}.json")
            with open(meta_path, 'w') as f:
                json.dump(info, f)

    def download_tags(
        self,
        tags,
        directory,
        save_image=True,
        save_metadata=True,
    ):
        query = self.query_post(tags)
        for info in self.logger.tqdm(query, desc=f"download {tags}"):
            self.download_one(info, directory=directory, save_image=save_image, save_metadata=save_metadata)


def download(url, path):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return path


def dec_to_base36(num):
    if num < 0:
        return '-' + dec_to_base36(-num)
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if num == 0:
        return "0"
    result = ""
    while num > 0:
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
