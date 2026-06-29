爬虫坏了不用重写！Scraper Studio Self-Healing 自愈功能演示

# 爬虫维护痛点


大家好，这期我们来测一个非常贴近开发者日常的问题：爬虫坏了怎么办？
做过爬虫维护的朋友应该都知道，真正麻烦的不是第一次把数据抓下来，而是后面网站一改版，原来的 CSS Selector 全部失效。比如以前商品标题在 .product-title，价格在 .price_color，结果页面结构一变，字段直接变成 undefined、空字符串，或者整条数据都没了。
传统做法是重新打开 DevTools，重新定位 DOM，改 selector，改解析逻辑，再重新测试、上线。这个过程很重复，而且维护成本很高。
今天我用 Bright Data 的 Scraper Studio 做一个技术测评，重点演示它的 Self-Healing Tool：当爬虫字段坏掉或者需要新增字段时，我们不用手写 CSS Selector，而是直接用自然语言告诉 AI：“这个字段抓不到了，帮我修复。”看它能不能自动重构 collector 代码。

---
# 测试网站和目标字段
这次我用一个电商的书店页面，有商品列表、分页、标题、价格、库存、评分、图片、详情页等字段，来演示电商爬虫的流程。
这里我的目标 URL 填的是：
https://books.toscrape.com/
最开始我让 Scraper Studio 生成一个基础 collector，字段包括：
title、product_detail_url、category、upc、product_type、tax、number_of_reviews、description。
等待一会后，就有了初始的模版。

---
# 当前 Collector 输出结构
这里是 Scraper Studio 生成出来的输出 schema。
可以看到它现在是一个 Object，里面已经定义了一些字段，比如：
title 是字符串，
price 可以设置成 Price 或 Money 类型，
availability 是字符串，
image_url 是 Image 或 URL 类型。
这个 schema 很重要，因为 Scraper Studio 的 collector 最终不是随便吐一段 HTML，而是要产出结构化数据。也就是说，最后可以导出 JSON、CSV，或者接到后面的数据管道里。

---
# 进入 Self-Healing / Refactor Collector 面板
如果后面产品经理或者数据团队说，我们还需要补三个字段：price、availability 和 image_url。这在实际爬虫项目里很常见，字段需求变了，或者网站改版后原字段抓不到了。
接下来我进入 Refactor collector，也就是 Self-Healing 的入口。
这里的说明写得很清楚：
Edit collector's code using AI for changing output fields or fixing a broken collector。
翻译过来就是：可以用 AI 来修改 collector 代码，适用于两类场景：
第一类，修改输出字段，比如我要新增 price、availability、image_url。
第二类，修复坏掉的 collector，比如字段返回空值，或者网站结构变化导致抓不到数据。
这里我勾选 Use custom input data，URL 还是填：
https://books.toscrape.com/
这样 AI 在 refactor 的时候就知道要基于这个实际目标页面来修复。


---
# 输入自然语言 Prompt，让 AI 新增字段
现在重点来了，我不手写 selector，直接在输入框里写自然语言需求。
我这里输入的是：
“Add three output fields to the collector: price, availability, and image_url. For each book on Books to Scrape, extract price from the book card, extract availability or stock status from the book card or detail page, and extract image_url as the absolute image URL, not a relative path. Keep the existing scraping logic unchanged. Continue following pagination across all book listing pages. If any field is missing, return null instead of breaking the collector.”
这段 prompt 有几个关键点。
第一，我明确告诉它要新增哪三个字段：price、availability、image_url。
第二，我说明每个字段从哪里来，比如 price 从商品卡片抓，availability 可以从卡片或详情页抓。
第三，我特别强调 image_url 要返回绝对 URL，不要返回相对路径。
第四，我要求保持现有逻辑不变，继续处理分页。
第五，如果字段缺失，返回 null，不要让整个 collector 报错。
这其实就是一个比较标准的 Self-Healing Prompt 写法：字段名要明确，错误现象要明确，期望输出要明确，容错策略也要明确。

---
# 等待 AI 生成代码 Diff，解释产品逻辑
提交之后，Scraper Studio 会开始 refactor collector。
这里我理解它的核心不是单纯问 AI：“帮我写个爬虫。”而是把当前 collector 的代码、输出 schema、目标 URL 和我们的自然语言需求结合起来，然后生成一个代码修改建议，也就是 diff。
这个设计对工程化很重要。因为生产环境里的爬虫通常不是一次性的脚本，而是一个长期维护的 collector。我们希望 AI 改的是局部逻辑，不要把原来能跑的分页、详情页解析全部推翻重来。
等它生成完成后，我们会看到一个代码 diff。这个时候不要盲目接受，要像 code review 一样看几个点：
第一，有没有真的新增 price、availability、image_url 三个字段。
第二，image_url 有没有从相对路径转成绝对路径。
第三，原来的 title、product_detail_url、详情页字段有没有被误删。
第四，遇到空字段时，是不是做了容错，而不是直接 throw error。
第五，分页逻辑有没有保留。
如果这些都没问题，再点击 Apply 或 Accept。

---
# Run Preview，对比修复前后结果
现在我接受 AI 生成的修改，然后跑一次 Preview。
大家看输出结果，这里每本书的数据里已经多了三个字段：
price：比如 £51.77；
availability：比如 In stock；
image_url：这里应该是完整图片地址，而不是 ../../media/cache/... 这种相对路径。
这一步就是 Self-Healing 的关键价值：
我们没有打开 DevTools，没有手写新的 CSS Selector，也没有手动改 parser，而是用自然语言描述需求，让 AI 自动重构 collector。
如果把这个场景换成真实业务，就是某电商网站改版后，价格字段突然返回 undefined。传统做法可能需要开发者排查 DOM、改代码、发版。而现在可以直接在 Self-Healing 里写：
“price 字段现在返回 undefined，请根据当前 HTML 修复价格解析逻辑，并保持输出 schema 不变。”
AI 会生成修复建议，我们 review diff，再预览结果。这个流程比从零重写爬虫轻很多。

---
# 技术测评总结：产品特点和适用边界
从技术测评角度看，我觉得 Scraper Studio Self-Healing 的核心特点有三个。
第一，它把爬虫维护从手写 selector 变成自然语言 refactor。这对经常维护爬虫的团队很有价值。
第二，它不是完全黑盒。AI 会生成代码 diff，开发者可以 review、accept、preview，再决定是否保存到生产环境。这比直接让 AI 自动上线安全很多。
第三，它适合字段级修复和 schema 调整。比如 price 抓不到了、title 变成 undefined、想新增 image_url、rating、availability，这些都是非常典型的自愈场景。
第四，Self-Healing 不是简单重新生成一个新爬虫，而是在现有 collector 上做 refactor，这更符合真实生产环境的维护方式。
当然，如果目标网站本身有复杂登录、强交互、动态渲染，或者字段来自接口而不是 HTML，就需要更清楚地告诉它数据来源，必要时还要切换 worker 类型或者配合浏览器交互逻辑。

---
4:50-5:10 结尾：强化降本增效
如果你只是偶尔写一次爬虫，可能感觉不到维护成本。但如果你负责的是长期运行的商品监控、价格监控、内容聚合、竞品数据采集，那么爬虫真正的成本一定在后期维护。
Scraper Studio 的 Self-Healing Tool 解决的正是这个痛点：
爬虫坏了，不一定要重写；字段失效，也不一定要手动改 selector。
你只需要用自然语言说清楚：哪个字段坏了、期望抓什么、输出格式是什么，AI 就可以帮你重构 collector 代码。开发者负责 review 和验证，这样既保留工程可控性，又明显降低维护成本。
这就是我这期对 Scraper Studio Self-Healing 的技术演示。对于有爬虫维护经验的开发者来说，这个功能确实值得重点关注。
视频评论区注册体验，有额外25美刀，欢迎尝试。


https://www.bright.cn/products/web-scraper/custom?utm_source=brand&utm_campaign=brnd-mkt_cn_csdn_wutong202606&promo=brd06
