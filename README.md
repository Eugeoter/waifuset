# Waifuset

Waifuset 是一个专门用于构建和管理文生图数据集的工具，其设计重点在于提供一个安全、高效、便捷、可以远程访问的用户界面（UI）。

![UI](/docs/assests/ui.jpeg)

## 特点

- **安全**：Waifuset 在操作过程中，不会修改数据集中的任何图像文件，确保数据的完整性和安全。
- **高效**：大部分编辑操作在 Waifuset 中都非常高效，能够轻松处理十万乃至百万量级的大型数据集。
- **易用**：UI 设计简洁直观，即使是复杂操作也可以轻松一键完成。
- **远程访问**：通过 Web 服务进行访问，支持在本地或云服务器上部署和访问。
- **UI 交互**：Waifuset 的所有操作都通过用户界面完成，即使是没有编程背景的用户也能轻松上手。

## 局限性

- **基于标签的标注**：Waifuset 假设所有的标注都是以英文逗号分隔的标签形式，而非自然语言描述。所有的标注编辑操作都将严格遵循这一格式。
- **专注于 Danbooru 标签**：Waifuset 的标注优化算法主要面向 [Danbooru 标签系统](https://danbooru.donmai.us/tags)，不支持其他类型的标签体系。
- **标注专用**：该工具主要用于编辑图像标注，而非图像本身。不包含图像编辑和文件系统功能。

## 适用场景

1. **数据集标注的构建和清洗**：Waifuset 特别适用于构建和维护文生图数据集的标注，提供了一种高效和简便的方式。
2. **数据集浏览**：作为一个数据集浏览器，Waifuset 可以帮助用户有效地查看和管理他们的数据集。

## 安装指南

为了使用 Waifuset，您需要首先安装项目及其依赖项。请按照以下步骤操作：

### 基本安装

1. 打开您的控制台。
2. 输入以下命令以克隆项目并安装所需的依赖项：

   ```bash
   git clone https://github.com/Eugeoter/waifuset
   cd waifuset
   pip install -r requirements.txt
   ```

### 可选安装

- **如需使用 WaifuScorer 进行美学评分**

  如果您希望使用 [WaifuScorer](https://huggingface.co/Eugeoter/waifu-scorer-v2) 进行美学评分，您需要安装额外的依赖项。在控制台中输入以下命令：

  ```bash
  pip install torch torchvision pytorch-lightning huggingface-hub git+https://github.com/openai/CLIP.git
  ```

- **如需使用 WaifuTaggerV3 进行图像标注**

  若要使用 [WaifuTaggerV3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) 对图像进行标注，您需要安装一些特定的依赖项。请在控制台中输入以下命令：

  ```bash
  pip install torch torchvision onnxruntime huggingface-hub
  ```

### 如何打开控制台

- **Windows**：
  [点此查看 Windows 控制台打开方法](https://blog.csdn.net/weixin_43131046/article/details/107030089)
- **MacOS**：
  [点此查看 MacOS 控制台打开方法](https://support.apple.com/zh-cn/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac)
- **Linux**：
  Linux 用户通常使用控制台进行操作。

## 使用指南

### 快速开始

Waifuset 的启动和使用非常直观。请遵循以下步骤快速开始：

1. **配置启动参数**

打开 launch_ui.py 文件，在 `get_config` 函数中通过修改变量值来配置您的启动参数。

```python
def get_config():
    dataset_source = r"/path/to/your/dataset/source"
    ... # 其它参数
```

如果您需要读取数据标注：

```python
def get_config():
    dataset_source = {
      "name_or_path": r"/path/to/your/dataset/source",
      "read_attrs": True,
    }
    ... # 其它参数
```

其中，`dataset_source` 为您的数据集源，其用法请参考[配置文件说明](docs/ui.md)的相关部分。

2. **启动 Waifuset**

   在控制台中，输入以下命令以启动 Waifuset：

   ```bash
   python launch_ui.py
   ```

3. **访问 UI 界面**

   - 在命令成功执行后，控制台会输出一个本地 URL，例如 `Running on local URL: http://xxx.xxx.xxx.xxx:xxxx`。
   - 打开浏览器并输入显示的 URL（即 `http://xxx.xxx.xxx.xxx:xxxx` 部分）以访问 Waifuset 的用户界面。

4. **云端服务器使用**

   - 如果您在云端服务器上部署 Waifuset，您可能需要将 `share` 参数设置为 `True` 以将 Gradio 网页共享到公网。
   - 或者，您可以使用服务器的代理功能访问 UI 界面。

按照这些步骤操作，您可以轻松地开始使用 Waifuset 来管理和编辑您的数据集。

### 用户手册

[点此查看完整用户手册](docs/ui.md)

### 如何更新

为了更新 Waifuset，请遵循以下步骤：

1. 打开您的控制台。
2. 进入项目目录：

   ```bash
   cd path/to/waifuset
   ```

   _注：请将 `path/to/waifuset` 替换为实际的项目路径。_

3. 执行以下命令以从远程仓库拉取最新的更新：

   ```bash
   git pull
   ```

   - 如果您之前修改过项目源码，例如更改过配置文件，这可能会导致冲突。
   - 若要强制覆盖本地修改并更新项目，请使用：

     ```bash
     git reset --hard
     git pull
     ```

### 安装到 pip

若您希望通过 pip 安装 Waifuset，以便在您的 Python 环境中使用其类和函数，您可以直接从 GitHub 安装。请确保您的环境中已安装 git。然后，运行以下命令：

```bash
pip install git+https://github.com/Eugeoter/waifuset.git
```

或者，在控制台中打开 git clone 后的项目目录，然后运行以下命令：

```bash
pip install .
```

这将允许您在 Python 脚本中导入和使用 Waifuset。
