
@[TOC](抓取ChatGPT大模型回答并做品牌曝光分析)

---






![请添加图片描述](https://i-blog.csdnimg.cn/direct/575e9d5073f44046949a6236cc1bbcf3.png#pic_center =100x100)

<center><font color="#007FFF" face="STKaiti" size="4">🌌你好！这里是 <a href="https://blog.csdn.net/WTYuong">晓雨的笔记本
 </a></center>
<center><font color="#007FFF" face="KaiTi"> 在所有感兴趣的领域扩展知识，感谢你的陪伴与支持~ </center>
<center><font color="#007FFF" face="KaiTi">👋 欢迎添加文末好友，不定期掉落福利资讯 </font></center>

---



# 写在最前面
>版权声明：本文为原创，遵循 [CC 4.0 BY-SA](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 协议。转载请注明出处。


[赠送25美金的注册链接](https://www.bright.cn/products/web-scraper/llm/?utm_source=brand&utm_campaign=brnd-mkt_cn_csdn_wutong202605&promo=brd05)





传统爬虫一般抓网页，比如电商价格、新闻内容、招聘信息。

但是现在很多用户已经开始直接问大模型，比如：

推荐几个适合中小企业的网络安全厂商。 哪个数据分析工具适合创业公司？ 哪些品牌适合做企业级安全建设？

这就产生了一个新的技术问题：

企业如何知道自己的品牌是否出现在大模型的推荐结果里？

所以我们这次设计一个小型商业化场景：

模拟一家 100 人规模的电商公司，向 ChatGPT 咨询网络安全厂商推荐。 然后我们用 Bright Data 的 ChatGPT Scraper 抓取回答，并分析目标品牌和竞品是否出现。

这是一个**AI 搜索可见性监测**场景。看看它能不能完成一个具体任务：

自动向 ChatGPT 提交问题，抓取回答结果，并把结果转成可以分析的数据。

我写一个工具代码已开源，适合用在 AI 搜索结果监测、品牌曝光分析、竞品分析、内容推荐监测等场景。

---

# 一、方案对比：为什么用 LLM Scraper？
在实现这个功能之前，其实有几种方案。

第一种是**人工提问**。 优点是最简单，直接打开 ChatGPT 问就行。 缺点是不能批量、不能自动化，也不适合长期监测。

第二种是**官方****大模型****API**。 比如直接调用 OpenAI API。 优点是稳定、接口规范。 但缺点是它测的是 API 模型回答，不一定等同于 chatgpt.com 网页端的实际表现。

第三种是**自己写浏览器自动化脚本**。 比如 Selenium 或 Playwright 操作 ChatGPT 页面。 优点是灵活。 缺点是开发成本高，需要处理登录、反爬、等待、页面结构变化等问题。

第四种就是这次测试的 **Bright Data ****LLM**** Scraper**。 它的特点是把网页端大模型交互封装成 API。我们只需要传入 prompt，就能拿到结构化返回结果。

所以从测评角度看，它的优势主要是：

接入成本低、返回结构化、适合批量测试和监测类任务。

# 二、接入方式
接下来进入技术实现。

Bright Data 的 ChatGPT Scraper 是通过 API 调用的。

核心接口是：

```plain
https://api.brightdata.com/datasets/v3/scrape
```

这里要传一个 `dataset_id`：

```plain
gd_m7aof0k82r803d5bjm
```

这个名字容易让人误会。 在这个 Demo 里，`dataset_id` 不是一个已经采好的静态数据集，而是用来指定要调用哪个 Scraper。 也就是说，这个 ID 对应的是 ChatGPT Scraper。

请求体里最关键的是 `prompt`：

```plain
payload = [
    {
        "url": "https://chatgpt.com/",
        "prompt": "一家100人规模的电商公司想采购网络安全服务，请推荐3个中国网络安全厂商，不超过80字。",
        "country": "",
        "web_search": False,
        "additional_prompt": ""
    }
]
```

这里 `web_search` 设置为 `False`。 这样做的原因是：这个 Demo 主要测试 ChatGPT Scraper 的基本抓取能力，不需要额外联网搜索，返回内容也会更短。

---

# 三、代码流程
代码整体分成四步。

第一步，输入 API Key。

第二步，构造请求头。

```plain
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}
```

这里要注意，`Authorization` 的格式必须是 `Bearer + 空格 + API Key`。

第三步，发送请求。

```plain
response = requests.post(
    api_url,
    headers=headers,
    json=payload,
    timeout=120
)
```

这里可能返回两种情况。

如果状态码是 `200`，说明结果已经直接返回。 如果状态码是 `202`，说明任务已经提交，但是结果还没准备好。

我这次测试时返回的就是 `202`，并且返回了一个 `snapshot_id`。

---

# 四、处理 202 Snapshot
`202` 在这里不是失败，而是 Bright Data 的异步结果机制。

返回内容里会有类似这样的字段：

```plain
{
  "snapshot_id": "sd_mpk14axy1nhs5b20s3",
  "message": "Your request is still in progress..."
}
```

接下来要用这个 `snapshot_id` 查询任务进度：

```plain
/datasets/v3/progress/{snapshot_id}
```

当状态变成：

```plain
{
  "status": "ready"
}
```

就可以下载结果。

这次测试里，返回状态是：

```plain
status: ready
records: 1
errors: 0
```

说明任务成功完成，并且抓取到 1 条结果。

---

# 五、返回字段解析
下载结果后，可以看到 Bright Data 返回了比较完整的结构化字段，包括：

`prompt`：原始问题。 `answer_text`：ChatGPT 的文本回答。 `answer_html`：HTML 格式回答。 `web_search_triggered`：是否触发了联网搜索。 `model`：使用的模型。 `citations` 和 `search_sources`：引用和搜索来源。

我在代码里做了兼容处理： 优先读取 `answer_text`，如果没有，再读取 `answer_text_markdown` 或 `answer_html`。

这样后面就可以把回答转成普通文本，然后做品牌关键词检测。

---

# 六、测试结果
这次 ChatGPT 返回的回答是：

推荐：奇安信、深信服、安恒信息。适合100人电商企业，覆盖终端、防火墙与数据安全。

然后我们用 Python 做了一个简单的品牌曝光分析。

结果如下：

| **<font style="color:rgb(0, 0, 0);">指标</font>** | **<font style="color:rgb(0, 0, 0);">结果</font>** |
| --- | --- |
| <font style="color:rgb(0, 0, 0);">目标品牌</font> | <font style="color:rgb(0, 0, 0);">奇安信</font> |
| <font style="color:rgb(0, 0, 0);">是否提到目标品牌</font> | <font style="color:rgb(0, 0, 0);">是</font> |
| <font style="color:rgb(0, 0, 0);">是否形成推荐</font> | <font style="color:rgb(0, 0, 0);">是</font> |
| <font style="color:rgb(0, 0, 0);">出现的竞品</font> | <font style="color:rgb(0, 0, 0);">深信服、安恒信息</font> |
| <font style="color:rgb(0, 0, 0);">竞品数量</font> | <font style="color:rgb(0, 0, 0);">2</font> |
| <font style="color:rgb(0, 0, 0);">是否触发联网搜索</font> | <font style="color:rgb(0, 0, 0);">FALSE</font> |
| <font style="color:rgb(0, 0, 0);">使用模型</font> | <font style="color:rgb(0, 0, 0);">gpt-5-5</font> |


这个结果说明，在当前 prompt 下，ChatGPT 推荐了目标品牌“奇安信”，同时也推荐了“深信服”和“安恒信息”两个竞品。

如果把这个测试扩展到更多 prompt，比如不同行业、不同预算、不同安全需求，就可以统计品牌在 AI 回答中的出现频率、推荐顺序和竞品共现情况。

---

# 七、技术测评总结
从这次实测来看，Bright Data 大模型 Scraper 的主要特点有几个。

第一，**接入门槛低**。 不需要自己写 Selenium 或 Playwright 去操作网页，也不用处理页面等待和结构变化。直接通过 API 提交 prompt 即可。

第二，**返回数据结构比较完整**。 除了回答文本，还包括 HTML、Markdown、模型、联网搜索状态、引用来源等字段，适合后续做数据分析。

第三，**适合批量监测场景**。 比如品牌曝光监测、竞品推荐分析、AI 搜索结果追踪，这些都可以通过批量 prompt 来做。

第四，**需要注意异步机制**。 实际调用时不一定马上返回结果，可能会返回 `202`。所以代码里最好同时处理 `200` 和 `202` 两种情况。

第五，**成本和稳定性需要控制**。 prompt 数量、输出长度、是否开启 web search，都会影响等待时间和成本。做 Demo 时可以先用短 prompt 和 `web_search=False`。

和自己写浏览器自动化相比，它的优势是省掉了很多维护成本；

和直接调用普通大模型 API 相比，它更适合测试网页端大模型搜索结果的抓取和结构化分析。

---

# 八、收尾
这次我们用一个最小 Demo 跑通了完整流程：

提交 prompt，处理 snapshot，下载结果，解析回答，生成品牌曝光分析表。

最终可以把 ChatGPT 的自然语言回答，变成一张可分析的数据表。

这就是本次 Bright Data ChatGPT Scraper 的技术实测。

欢迎大家注册亮数据体验，评论区有额外的25美刀试用金，欢迎体验！

我们下次再见！


---

>hello，这里是 <a href="https://blog.csdn.net/WTYuong?spm=1010.2135.3001.5343">晓雨的笔记本 </a>。如果你喜欢我的文章，欢迎三连给我鼓励和支持：👍点赞  📁 关注 💬评论，我会给大家带来更多有用有趣的文章。
原文链接 👉  ，⚡️更新更及时。

欢迎大家点开下面名片，添加好友交流。


