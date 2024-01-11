# 1. 介绍

该项目是一个用于二次元图像-文本标注管理和优化的工具，并提供 UI 交互界面。

# 2. 安装

1. `git clone https://github.com/Eugeoter/sd-dataset-manager`
2. `cd sd-dataset-manager`
3. `pip install -r requirements.txt`

# 3. 使用方法

## 3.1. UI

使用命令行参数启动 UI 界面。

### 3.1.1. 启动参数

#### 用法 1

`python api.py --source 'path/to/folder' --write_to_txt`

其中，`--source` 参数指定了数据集的根目录，`--write_to_txt` 参数指定了是否将标注结果写入到图像文件同名的 txt 文件中，这是常见的标注方式。

#### 用法 2

`python api.py --source 'path/to/folder_or_database' --write_to_database --database_file 'path/to/database.json'`

其中，`--source` 参数指定了数据集的根目录**或者**数据库文件，`--write_to_database` 参数指定了是否将标注结果写入到数据库文件中，而 `--database_file` 指定了数据库文件的输出/保存位置，这是一种更加灵活的标注方式。

对于将结果写入到您可以不指定，指定一种，或都指定。如果您不指定，那么所有数据集操作都将是模拟操作，不会对任何文件产生影响。如果您都指定，那么使用时保存数据库时将会同时写入 txt 文件。

#### 其他参数

`--formalize_caption`: 是否在加载时将标注格式化。

`--subset_chunk_size`: 每页显示的图像数量，默认为 80。太高可能会导致加载缓慢。

`--share`: 是否共享 Gradio 网页。

### 3.1.2. 界面操作

...
