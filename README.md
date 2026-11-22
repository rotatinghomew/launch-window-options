# 发射观礼行程延期时的组合保障与选择权估值

窗口变化时核对观礼及普通游线的可选权利和实际费用。

## 资料和运行环境

`data/` 提供本地合成业务资料。各业务文件为 UTF-8 JSON 数组，键、单位和关联见 [资料字典](data/README.md)。`format.json` 记录列集合和基础引用约束。

使用 Python 3.11。在普通 Python 环境可用 `python3 -m pip install -r requirements.txt` 安装锁定的分析依赖。检查提供的资料可运行：

```sh
python3 -m unittest discover -s tests -v
```

资料检查覆盖文件可读性、字段与明确的单列引用。业务原始记录允许保留冲突、未决关系及尚未到达的观察。
