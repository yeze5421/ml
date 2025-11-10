
# 一键刷题 · 静态版

- 打开 `index.html` 即可使用（本地或放到任意静态网站/GitHub Pages）。
- 已内置：**XGBoost 基础 100 题**（点击即刷）。
- 可通过 **上传本地 JSON** 或 **粘贴 URL** 导入新的题库（一次只选一个）。

## 快速用法
1. 双击 `index.html` 本地打开（或部署到任意静态服务器）。
2. 在首页：
   - 直接点击「XGBoost 基础 100 题」开始；
   - 或点击「导入 JSON」上传一个题库；
   - 或在「通过 URL 导入」里粘贴一个在线 JSON 直链。
3. 开始后即可答题、查看解析、保存本地进度。

## 添加更多内置题库（可选）
- 把新的 JSON 文件放进 `banks/` 目录；
- 编辑 `banks.json`，新增一条记录，`key/title/path/format` 填好即可；
- 重新打开页面。

## JSON 说明
- 本仓库内置的 `banks/xgboost_basic_100.json` 为 `xgb-basic-v1` 格式：
  ```json
  {
    "id": "xgb-001",
    "type": "single" | "multi" | "short",
    "stem": "题干",
    "options": ["A", "B", "C"],                // 简答题可无
    "answer": [0, 1],                          // 选项下标数组；简答题可用字符串或字符串数组
    "explain": "解析"
  }
  ```
- 如果你的题库字段不同，可以先转换为上面格式，或在页面中用 URL 导入时选择「格式=自动」并提供示例字段名。

