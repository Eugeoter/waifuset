import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='API arguments parser')

    parser.add_argument('--source', type=str, required=True, help='Dataset source, can be a directory root, a json file or a csv file / 数据集源，可以是一个数据集文件夹，一个json文件或一个csv文件')
    parser.add_argument('--write_to_txt', action='store_true', help='Whether to write to txt caption files `{image}.txt` when saving / 是否在保存时将结果写入 `{图像}.txt` 的标注文件')
    parser.add_argument('--write_to_database', action='store_true', help='Whether to write to database when saving / 是否在保存时将结果写入数据库')
    parser.add_argument('--database_file', type=str, required=True, help='Database file output path / 数据库文件的输出路径')

    parser.add_argument('--formalize_caption', action='store_true', help='Whether to formalize caption when loading / 是否在加载时将标注格式化')
    parser.add_argument('--change_source', action='store_true', help='Whether to change image source when loading / 是否在加载数据源时替换数据图像的来源')
    parser.add_argument('--old_source', type=str, help='Old image source / 旧的图像来源')
    parser.add_argument('--new_source', type=str, help='New image source / 新的图像来源')

    parser.add_argument('--share', action='store_true', help='Whether to share the API / 是否共享API')

    args = parser.parse_args()
    return args
