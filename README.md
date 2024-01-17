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

`python api.py --source 'path/to/folder' --write_to_txt --language cn`

其中，`--source` 参数指定了数据集的根目录，`--write_to_txt` 参数指定了是否将标注结果写入到图像文件同名的 txt 文件中，这是常见的标注方式。

#### 用法 2

`python api.py --source 'path/to/folder_or_database' --write_to_database --database_file 'path/to/database.json' --language cn`

其中，`--source` 参数指定了数据集的根目录**或者**数据库文件，`--write_to_database` 参数指定了是否将标注结果写入到数据库文件中，而 `--database_file` 指定了数据库文件的输出/保存位置，这是一种更加灵活的标注方式。

对于将结果写入到您可以不指定，指定一种，或都指定。如果您不指定，那么所有数据集操作都将是模拟操作，不会对任何文件产生影响。如果您都指定，那么使用时保存数据库时将会同时写入 txt 文件。

#### 其他参数

`--formalize_caption`: 是否在加载时将标注格式化。

`--chunk_size`: 每页显示的图像数量，默认为 80。太高可能会导致加载缓慢。

`--share`: 是否共享 Gradio 网页。

`--port`: Gradio 网页的端口。

`--language`: 界面语言，默认为英文。从 `en` 和 `cn` 中选择。注意：无论选择何种语言，日志信息均为英文。

### 3.1.2. 界面操作

#### 数据解界面

UI 的数据集界面大体由两个部分组成：左侧的图像显示区域和右侧的标注处理区域。

单击左侧画廊中的图像即选中数据，选中数据后，其他面板将显示该数据的详细信息，并允许对其进行标注处理。

![Alt text](/help/2.png)

按键说明

1. 子集刷新按钮：重新加载 4. 所选中的子集；
2. 切换至上一个页面；
3. 切换至下一个页面；
4. 子集和页面选择：子集列表和画廊页面选择，可通过更改其值来加载特定的子集及其页面；
5. 日志：显示当前的操作日志；
6. 切换至随机历史记录的上一个图像；
7. 切换至随机历史记录的下一个图像；
8. 刷新画廊；
9. 删除选中的数据。不会实际删除文件：当启用 --write_to_database 时，会从数据库中移除该数据；当启用 --write_to_txt 时，会重命名数据图像和标注文件为原名+`.bak`，表示备份，以便后续恢复。
10. 标注格：显示当前图像的标注信息，可任意编辑。其中的标注按照英文逗号`,`分割，将自动维持标注格式。
11. 随机采样：从当前子集中随机采样一个图像，显示在图像显示区域。采样的图像不会重复，直到采样完毕。
12. 显示所选数据所在子集；
13. 撤销：撤销上一次对该数据的标注编译操作；
14. 反撤销：恢复上一次对该数据的标注编译操作；
15. 保存：保存并实际写入至今为止对数据集的全部修改。当启用 --write_to_database 时，会将修改写入数据库；当启用 --write_to_txt 时，会将修改写入 txt 文件。只有当点击保存按钮时，才会真正写入，否则所有修改都为草稿操作，不会实际生效。
16. 批处理：开启或关闭批处理选项。当启用时，除直接编辑标注以外的所有标注编辑操作将会对整个所选中的子集生效。
17. 所选图像路径；
18. 打开所选图像所在位置；
19. 所选图像的分辨率；
20. 加载或刷新数据集中所有出现过的标签到 24. 和 27. 中，方便选取；
21. 卸载由 20. 加载的标签，以防止过多标签造成卡顿；
22. 开启或关闭正则表达式匹配。当启用时，
23. 查询标签 24. 的条件。“任意”表示只要任意 24. 中的标签被包含某个数据的标注中，那么它就会被匹配；“全部”表示只有当 24. 中的标签全部被包含在标注中时，才会被匹配；
24. 查询结果中包含的标签；
25. 连接包含和排除的标签。“且”表示对包含和排除的匹配项取交集；“或”表示对包含和排除的匹配项取并集；
26. 查询标签 27. 的条件。与 23. 相同；
27. 查询结果中排除的标签；
28. 查询按钮：根据 23. - 27. 的条件进行查询，将查询结果显示在画廊中；

##### 快速标注界面

![Alt text](/help/3.png)

快速标注界面用于快速为选中数据添加特定标签。在上图的界面中，各按键为预设的标签，从左至右，从上至下依次为：

best quality, high quality, low quality, worst quality

beautiful color, detailed, lowres, messy

aesthetic, beautiful

##### 自定义标注界面

![Alt text](/help/4.png)

自定义标注界面用于快速为选中的数据添加或删除自定义标签。在上图的界面中，每一行表示一个或多个自定义标签，每一行的加号表示添加该标签，减号表示删除该标签。

##### 高级标注界面

![Alt text](/help/5.png)

高级标注界面用于快速为选中的数据以自定义地方式添加或删除自定义标签。该界面的操作方式为：

如果 `任意/全部` 标签 `Y` 都被 `包含/排除`，则 `添加/移除` 标签 `X`。

其中，标签 `X` 为第一行的标签，标签 `Y` 为第二行的标签。

##### 标注优化界面

![Alt text](/help/6.png)

标注优化界面用于一键调整选中数据的标注。几个按钮的调整方式如下：

1. 格式化：将标签格式化为以下形式：

- 替换下换线为空格：`x_y_z` -> `x y z`，x, y, z 为任意字符串；
- 为括号添加转移符：`x (y)` -> `x \(y\)`，x, y 为任意字符串；
- 自动提取艺术家、角色和风格标签，并添加注释：` by xxx` -> `artist: xxx`，`ganyu \(genshin impact\)` -> `character: ganyu \(genshin impact\)`，`realistic` -> `style: realistic`；

2. 排序：按照 NovelAI V3.0 的排序方案对标注中的标签排序；

3. 去重：去除标注中的重复标签；

4. 去重叠：去除标注中具有重叠语义的标签，并保留语义最丰富的一者。例如，`white shirt` 和 `shirt` 同时存在时，将会删除 `shirt` 而保留 `white shirt`；

5. 去特征：对于标注中包含角色标签的标注，去除该标注中所有与人物特征有关，而且是该角色的核心标签的标签。角色核心标签将由以下方式确定：
   统计整个大数据集中，所有标注中的角色标签的词频表，将出现次数多的标签作为该角色的核心标签。例如，数据集中由 100 个带有 `ganyu \(genshin impact\)` 的数据，其中 `blue hair` 出现了 100 次，`horns` 出现了 50 次，`red hair` 出现了 10 次，那么对于甘雨这个角色来说，`blue hair` 的词频为 1.0，`horns` 为 0.5，`red hair` 为 0.1。
   之后，删除所有这些包含 `ganyu \(genshin impact\)` 的数据中，词频大于等于某个值的标签。界面中的阈值滑条用于控制去除词频大于等于该阈值的标签。例如，将阈值设置为 0.5，那么 `blue hair` 和 `horns` 将会被去除，而 `red hair` 将会被保留。
   另外注意，当数据集中的数据量较少时，该功能可能会失效，因为词频统计的样本量太少，无法准确得知哪些标签是核心标签。

##### WD14 界面

![Alt text](/help/7.png)

WD14 界面用于快速为选中的数据使用 WD14 标注模型打标。默认使用的模型是 wd14 swinv2（https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2）。

该界面中，阈值和角色阈值滑条分别指定了一般标签和角色标签的最低置信度，低于该值的标签将会被去除。
覆盖模式指定了处理已有标注数据的方式，“覆盖”将直接覆盖已有标注，而“追加”和“前置”将在已有标注的基础上追加新的标注；分别位于已有标注的后面和前面；“忽略”将会仅对没有标注的数据进行标注，而不会对已有标注的数据进行处理。
