# DeepSeek-Learn 示例

本目录包含DeepSeek模型的使用示例，帮助您学习如何与大型语言模型交互。

## 示例列表

### simple_chat.py
一个简单的交互式对话示例，展示如何加载本地DeepSeek模型并进行对话。

#### 使用方法
1. 确保您已经下载了模型（使用 `util/download_model.py` 脚本）
2. 安装所有依赖：`pip install -r requirements.txt`
3. 运行对话示例：`python examples/simple_chat.py`

#### 学习要点
- 如何加载和初始化本地模型
- 如何构建对话提示（prompt）
- 如何使用流式输出
- 如何管理对话历史
- 如何处理用户输入

## 自定义模型路径
默认情况下，示例程序会使用配置文件中指定的默认模型。如果您想使用其他模型，可以修改 `config/config.py` 中的 `DEFAULT_MODEL_NAME` 参数。 