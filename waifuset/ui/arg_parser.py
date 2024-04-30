import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='API arguments parser')

    parser.add_argument('--source', type=str, default=None, nargs='+', help='Dataset source, can be a directory root, a json file or a csv file / 数据集源，可以是一个数据集文件夹，一个json文件或一个csv文件')
    parser.add_argument('--write_to_txt', action='store_true', help='Whether to write to txt caption files `{image}.txt` when saving / 是否在保存时将结果写入 `{图像}.txt` 的标注文件')
    parser.add_argument('--write_to_database', action='store_true', help='Whether to write to database when saving / 是否在保存时将结果写入数据库')
    parser.add_argument('--database_file', type=str, help='Database file output path / 数据库文件的输出路径')

    parser.add_argument('--share', action='store_true', help='Whether to share the API / 是否共享API')
    parser.add_argument('--port', type=int, help='Port to run the API / 运行API的端口')

    parser.add_argument('--language', type=str, default='en', help='Language of the UI / UI的语言，从 `en` 和 `cn` 中选择')
    parser.add_argument('--max_workers', type=int, default=1, help='Max workers when processing captions / 处理标注时的最大工作线程数')
    parser.add_argument('--chunk_size', type=int, default=80, help='Maximum number of displayed images in a gallery page / 单页画廊中显示单页显示的最大图像数')
    parser.add_argument('--render', type=str, default='full', help='Render mode of the UI / UI的渲染模式，从 `full` 和 `partial` 中选择')

    args = parser.parse_args()
    return args
