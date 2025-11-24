---
title: 'MCP'
pubDate: 2000-12-01
description: 'MCP'
heroImage: 'https://wolves.top/pic/mcp-1.png'
---

# MCP
MCP全称Model-Controller-Processor，是一个开放协议，使LLM应用程序能够与外部数据源和工具无缝集成。无论你是构建AI驱动的IDE、增强聊天界面，还是创建自定义AI工作流，MCP都提供了一种标准化方式，将LLM与它们所需的上下文连接起来。
[MCP](https://github.com/modelcontextprotocol)
其中的servers仓库中收录了大量好用的mcp工具

# 我的常用MCP
## 1. [Chrome DevTools MCP](https://github.com/ChromeDevTools/chrome-devtools-mcp)
- 自动抓取页面元素、执行点击、输入等操作  
- 实时获取页面性能数据与控制台日志  
- 支持断点调试与网络请求拦截

<details>
  <summary>GLOBAL</summary>

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "command": "npx",
      "args": ["-y", "chrome-devtools-mcp@latest"]
    }
  }
}
```
</details>

<details>
  <summary>codex</summary>

```bash
codex mcp add chrome-devtools -- npx chrome-devtools-mcp@latest
```

</details>

## 2. [context7](https://github.com/upstash/context7)
- 提供上下文管理功能，允许LLM应用程序在处理用户查询时访问和利用外部知识
- 支持实时数据更新，确保LLM应用程序始终使用最新信息
- 高度可定制化，可根据具体需求进行配置和扩展

<details>
  <summary>GLOBAL</summary>

```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp",
      "headers": {
        "CONTEXT7_API_KEY": "YOUR_API_KEY"
      }
    }
  }
}
```
</details>

<details>
  <summary>codex</summary>

```bash
[mcp_servers.context7]
url = "https://mcp.context7.com/mcp"
http_headers = { "CONTEXT7_API_KEY" = "YOUR_API_KEY" }
```
</details>
