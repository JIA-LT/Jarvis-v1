# DL-0010：Chunking 方案选型

## 背景

RAG 检索稳定性不足：使用通用按字符/token 切分时，文档稍有改动（如增删一行）会导致块边界漂移，进而影响 chunk_id、向量与溯源，召回与引用不稳定。

## 目标

- **稳定**：同一文档多次分块结果一致；小幅编辑后，未改动区的 chunk 边界与 ID 可复现。
- **可解释**：每个 chunk 具备语义上下文（如所属章节）与源码位置（行号），便于检索溯源与 Debug。

## 方案对比

| 维度 | 方案 A：自研 Markdown chunking | 方案 B：LangChain RecursiveCharacterTextSplitter |
|------|-------------------------------|--------------------------------------------------|
| 输入 | Markdown 全文 | 任意文本 |
| 输出 | `List[Dict]`：`chunk_id`、`content`、`header_path`、`start_line`、`end_line` | `List[str]`：仅文本块 |
| 边界 | 按标题层级 + 段落边界，可控、可复现 | 按字符/递归分隔符，与文档结构解耦 |
| 代码块 | 可配置保持 fenced 代码块完整 | 可能从代码块中间切开 |
| 依赖 | 无（仅 stdlib + 项目内） | LangChain |
| 实现与调参 | 需自研与迭代 | 开箱即用，调 `chunk_size` / `overlap` 即可 |

### Trade-off

- **可控性 vs 上手速度**：A 需投入实现与测试，但对 MD 结构、ID、行号、代码块等有完整控制；B 接入快，但难以做稳定 ID、header_path、行号追踪。

## 选择

**方案 A：自研 Markdown chunking**

## 理由

1. **稳定 ID**：`chunk_id` 由 `header_path + 行号区间 + 内容哈希` 生成，相同内容与结构即相同 ID，利于去重、更新与比对。
2. **header_path**：维护 `H1 > H2 > H3` 式标题路径，检索与展示时能直接看到 chunk 所属章节，提升可解释性。
3. **可回放**：`start_line` / `end_line` 可映射回源文件，便于定位、Diff 与回归测试（如用 `example.md` 做 chunks 逻辑验证）。

## 实现概要

- **实现位置**：`app/core/chunking.py`
- **入口**：`chunk_text(md_text, **kwargs) -> List[Dict]`

### 三阶段流程

1. **Pass 1：按标题切 section**  
   按 `#`–`######` 将 MD 切成大块；维护 `heading_stack`，构建 `header_path`；代码块内（三重反引号围起的部分）不识别标题。

2. **Pass 2：section 内按段落与大小切 chunk**  
   以空行为段落边界（代码块内不切），将段落合并为若干 **unit**；再按 `max_chars` 将 units 合并为 chunk；超过 `max_chars` 的单个 unit（如大代码块）单独成块。

3. **Pass 3：合并过小 chunk**  
   同 `header_path` 下，若相邻 chunk 小于 `min_chars`，则尝试合并；合并后不超过 `max_chars`。

### 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_chars` | 1800 | 单 chunk 最大字符数 |
| `min_chars` | 250 | 小于此值会尝试与相邻 chunk 合并 |
| `max_heading_level_for_path` | 3 | `header_path` 最多包含 H1–H3 |
| `keep_code_blocks_intact` | True | 保持 fenced 代码块完整 |

### Chunk 输出格式

```python
{
    "chunk_id": str,      # 稳定哈希 ID
    "content": str,       # 块文本
    "header_path": str,   # 如 "H1 引言 > H2 方法"
    "start_line": int,    # 起始行（1-based，含）
    "end_line": int,      # 结束行（1-based，含）
}
```

### 测试方式

使用 `data/md/example.md` 验证 chunks 逻辑：

```bash
python scripts/ingest_folder.py data/md
```

可检查：块数、`header_path`、行号区间、代码块是否完整、块大小是否落在 `min_chars`–`max_chars` 预期内。

## 责任（DRI）

你

## 兜底

若自研实现维护成本过高或遇无法解决的边界 case，可切回 **方案 B**（`app/core/chunking_langchain`），仅用其做“纯文本切块”，放弃稳定 ID / header_path / 行号等能力。

## 预期

- 相同 MD 多次分块结果一致，小幅修改仅影响变动区 chunk。
- 召回与引用具备 `header_path`、行号等元数据，检索稳定性与可解释性提升。
