# DeepSeek-Learn

这是一个用于学习和使用DeepSeek大型语言模型的项目。本项目提供了简单易用的工具和示例，帮助您快速上手使用DeepSeek系列模型。

## 项目结构

```
deepSeek-learn/
├── examples/          # 示例代码
│   ├── simple_chat.py # 简单对话示例
│   └── README.md      # 示例使用说明
└── README.md          # 项目说明文档
```

## 功能特点

- 提供简单的交互式对话接口
- 支持流式输出，实时显示模型回复
- 管理对话历史，支持多轮对话
- 详细的代码注释，方便学习和二次开发

## 使用方法

1. 确保您已经下载了DeepSeek模型（使用项目根目录下的 `util/download_model.py` 脚本）
2. 安装所有依赖：`pip install -r requirements.txt`
3. 运行对话示例：`python deepSeek-learn/examples/simple_chat.py`

## 学习资源

- [DeepSeek官方文档](https://github.com/deepseek-ai/)
- 项目根目录下的 `examples` 目录包含了更多使用示例
- 每个示例都有详细的代码注释，可以作为学习使用大语言模型的参考

## 自定义配置

您可以通过修改项目根目录下的 `config/config.py` 文件来自定义模型配置。 