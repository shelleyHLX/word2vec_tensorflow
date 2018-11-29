## 代码
An tensorflow implementation of Word2Vec model: Skip-gram, CBOW.
This implementation is based on [deep-learning--ud730](https://cn.udacity.com/course/deep-learning--ud730).
## 实验结果
```
停用词读取完毕，共1893个词
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.693 seconds.
Prefix dict has been built succesfully.
Data size 3427356
Most common words (+UNK) [['UNK', 4451], ('，', 389745), ('的', 180001), ('。', 65072), ('了', 63401)]
Sample data [43675, 40406, 13483, 7605, 0, 30279, 493, 2, 2199, 30279]
data: ['\ufeff', '《', '斗破', '苍穹', 'UNK', '第一章', '陨落', '的']

with half_window_size = 1:
    batch: [['\ufeff', '斗破'], ['《', '苍穹'], ['斗破', 'UNK'], ['苍穹', '第一章'], ['UNK', '陨落'], ['第一章', '的'], ['陨落', '天才'], ['的', '第一章']]
    labels: ['《', '斗破', '苍穹', 'UNK', '第一章', '陨落', '的', '天才']

with half_window_size = 2:
    batch: [['天才', '第一章', '的', '天才'], ['第一章', '陨落', '天才', '('], ['陨落', '的', '(', '本章'], ['的', '天才', '本章', '免费'], ['天才', '(', '免费', ')'], ['(', '本章', ')', '“'], ['本章', '免费', '“', '斗之力'], ['免费', ')', '斗之力', '，']]
    labels: ['陨落', '的', '天才', '(', '本章', '免费', ')', '“']

Average loss at step 0: 7.871643
Nearest to 都: 花蛇儿, 叛宗, 惨痛教训, 没耐, 岚, 四百六十章, 高一, 这两项,
Nearest to 后: 大起大落, 偶像, 地址, 妖之愧, 应话, 逆水行舟, 饮酒作乐, 我冰,
Nearest to 着: 死死, 这苏千, 章米, 吵闹声, 看清, 火莲时, 望, 抛开,
Nearest to 等: 烟雾, 师都会, 轻吻, 纹, 反增, 杨, 虚影, 戏谑,
Nearest to 与: 看夭夜, 所教, 来而不往非礼也, 买个, 男伴, 衍变, 真成了, 简中,
Nearest to 从: 小摊, 胸大, 地天, 一瘪, 小辈, 二十多年, 冒汗, 根源,
Nearest to 去: 吞纳下, 地字房, 通玄, 主一, 连斗王, 双头蛇, 虎鹰, 一千六百,
Nearest to 火焰: 资源库, 变身, 风筝, 其血, 命之时, 心惊, 张牙舞爪, 池中之物,
Nearest to 一道: 严加, 五星, 分毫, 云岚宗为, 卡岗疑, 局促, 击成, 毒物,
Nearest to ”: 银线, 突然行动, 那一, 昏头, 岩蛇, 轻吐, 骄, 筋疲力尽,
Nearest to /: 慈善, 白光, 魂镜握, 天际, 犹疑, 名下, 消除隐患, 求生,
Nearest to 之中: 说到做到, 进出口, 恶作剧, 如欣, 一代, 濒临, 顽, ，,
Nearest to 一个: 迎客, 名魂, 以切, 找死, 多时, 自墨, 从辰, 白脸,
Nearest to 来: 蓝炎, 没白上, 那连纳兰, 狮吼, 常态, 会争, 翻天覆地, 野心勃勃,
Nearest to 什么: 复苏, 之大, 尖子生, 本院, 刑罚, 预感, 鼎暴, 浑身上下,
Nearest to 能量: 很长, 他妈, 绕绕, 即可, 医仙望, 如便, 火轮, 万雷噬,
Average loss at step 2000: 3.660892
Average loss at step 4000: 2.753446
Average loss at step 6000: 2.535429
Average loss at step 8000: 2.595507
Average loss at step 10000: 2.451094
Nearest to 都: 所有人, 大多, 两人, 能, 会, 必须, 每天, ，,
Nearest to 后: 片刻, 完毕, 不久, 许久, 十分钟, 十几分钟, 左右, 小时,
Nearest to 着: 带, 对面, 盯, 握, 姚盛, 苏千, 挂, 对,
Nearest to 等: 修崖, 柳擎, 吴昊, 萧炎, 严皓, 着, 姚盛, 实力,
Nearest to 与: 柳擎, 吴昊, 琥嘉, 白程, 修岩, 期盼, 斗王, 凝重,
Nearest to 从: 至, 指间, 柳擎, 远处, 大赛, 其, 姚盛, 缓缓,
Nearest to 去: 找, 哪里, 散, 抹, 搽, 而, 投, 了,
Nearest to 火焰: 实质, 般的, 般, 无形, 犹如, 从, 的, 之中,
Nearest to 一道: 划起, 足有, 白影, 身影, 念头, 人影, 残影, 闪过,
Nearest to ”: 。, “, 凌影, 姚盛, ？, 闻言, ！, 大赛,
Nearest to /: /, com, :, 第四, 第三, www, 第一百, 第一百七,
Nearest to 之中: 沼泽, 内院, 凹槽, 斗晶, 场地, 密室, 经脉, 比赛,
Nearest to 一个: 每, 不慎, 这么, 小小的, 极为, 有着, 懒腰, 以,
Nearest to 来: 伸出手, 十, 分钟, 找, 话, 而, 转过身, 睁开眼,
Nearest to 什么: 东西, 发生, 造成, 好, ？, 有, 说, 了,
Nearest to 能量: 天地, 波动, 庞大, 精纯, 罩, 壁, 狂暴, 雄浑,
Average loss at step 12000: 2.374339
Average loss at step 14000: 2.261271
Average loss at step 16000: 2.250934
Average loss at step 18000: 2.126321
Average loss at step 20000: 2.091706
Nearest to 都: 大多, 怕, 一直, 会, 大家, 不是, ，, 谁,
Nearest to 后: 半晌, 十几分钟, ，, 一会, 几分钟, 结束, 瞬息, 落下,
Nearest to 着: 带, 望, 对, 盯, 握, 面对, 噙, 抓,
Nearest to 等: 萧炎, 静, 阵容, 宋清, 丘陵, 这, 宝贝, 人物,
Nearest to 与: 萧炎, 宋清, 曹家, 人, 丘陵, 老师, 叶重, 他,
Nearest to 从: 其, 天空, 他, 当日, 其中, 那, 便是, 它们,
Nearest to 去: 涌, 抹, 落, 而, 抓, 管, 投, 赶,
Nearest to 火焰: 柱, 无形, 的, ，, 壁, 熊熊, 两种, 罩,
Nearest to 一道: 发出, 划起, 化为, 白影, 黑影, 冷喝, 清脆, 每,
Nearest to ”: ！, 易尘, ？, 闻言, 。, 萧炎, 听得, “,
Nearest to /: /, 第九, com, 第百, 第, :, 第七, www,
Nearest to 之中: 森林, 院落, ，, 山谷, 木盆, 云层, 祭坛, 山脉,
Nearest to 一个: 约莫, 另外, 不慎, 小时, 将近, 小小的, 某, 形成,
Nearest to 来: 十, 试试, 分钟, 涌, 睁开眼, 以此, 扑, 敢,
Nearest to 什么: 时候, 可不是, 不对劲, 本事, 担心, 做, 简单, 不是,
Nearest to 能量: 磅礴, 涟漪, 壁, 狂暴, 膜, 波动, 天地, 匹练,
Average loss at step 22000: 2.017441
Average loss at step 24000: 1.981102
Average loss at step 26000: 1.956878
Average loss at step 28000: 2.025509
Average loss at step 30000: 1.932720
Nearest to 都: 大多, 会, 不敢, ，, 没, 每次, 他们, 是从,
Nearest to 后: 半晌, 不久, ，, 两天, 一会, 结束, 回去, 无果,
Nearest to 着: 盯, 对, 带, 抱, 萧炎望, 携带, 顶, 噙,
Nearest to 等: 萧鼎, 我, 你, 她, 这, 异火, 他们, 到时候,
Nearest to 与: 萧厉, 他们, 她, 二哥, 奥托, 柳席, 其他, 我,
Nearest to 从: 忽然, 萧鼎, 外面, 一旁, 悄悄的, 高空, 穆蛇, 自己,
Nearest to 去: 而, 寻找, 看看, 脱, 散, 先, 那里, 哪,
Nearest to 火焰: 柱, 熊熊, 蜥蜴人, 之中, 粉红, 的, 熊熊燃烧, 地,
Nearest to 一道: 闪过, 黑影, 流光, 划起, 发出, 残魂, 化为, 随着,
Nearest to ”: 闻言, “, …, 听得, ！, 青鳞, ？, 古河,
Nearest to /: :, /, 第百, www, 第一百, 第九, 第七, com,
Nearest to 之中: 帐篷, 眼瞳, 木盆, 沙漠, 黄沙, ，, 密林, 城市,
Nearest to 一个: 另外, 不慎, 懒腰, 小小的, 了, 接, 犹如, 人,
Nearest to 来: 扑, 寻找, 考核, 伸出手, 找, 转过身, 袭, 形容,
Nearest to 什么: 东西, 算不得, 事, 时候, 有, 这是, 搞, ？,
Nearest to 能量: 巨蛇, 膜, 狂暴, 纱衣, 天地, 浩瀚, 核, 波动,
Average loss at step 32000: 1.813917
Average loss at step 34000: 1.877401
Average loss at step 36000: 1.885831
Average loss at step 38000: 1.853735
Average loss at step 40000: 1.820487
Nearest to 都: 谁, 是, 所有人, 不敢, 未曾, 只能, 是因为, 会,
Nearest to 后: 半晌, ，, 十几分钟, 出来, 向, 瞬息, 三年, 出去,
Nearest to 着: 带, 对, 借助, 盯, 连带, 噙, 拥有, 紧咬,
Nearest to 等: 人, 静, 萧鼎, 下次, 我, 这, 明日, 木辰,
Nearest to 与: 慕兰谷, 炎盟, 蛇人族, 金雁宗, 法犸, 美杜莎, 萧炎, 能,
Nearest to 从: 指间, 至, 缓缓的, 何处, 远处, 各处, 那, 窗户,
Nearest to 去: 涌, 抹, 哪, 脱, 找, 抓, 寻找, 办,
Nearest to 火焰: ，, 柱, 森白, 熊熊, 的, 小蛇, 青白, 壁,
Nearest to 一道: 黑影, 发出, 化为, 划起, 影子, 掠过, 嘹亮, 人影,
Nearest to ”: “, ！, 。, 听得, ？, 小医仙, 啊, 语罢,
Nearest to /: 第七, 第, :, 第六, www, 第百, com, 第四,
Nearest to 之中: 脑海, 密室, 视野, 山谷, 沼泽, ，, 大殿, 森林,
Nearest to 一个: 小小的, 某, 小时, 角落, 懒腰, 不慎, 这么, 仅仅,
Nearest to 来: 转过身, 袭, 找, 而, ，, 分钟, 日, 十,
Nearest to 什么: 时候, 造成, 办法, 意外, 做, 东西, 算不得, 问题,
Nearest to 能量: 天地, 膜, 七彩, 球, 波动, 匹练, 黑暗, 漩涡,
Average loss at step 42000: 1.770576
Average loss at step 44000: 1.777092
Average loss at step 46000: 1.715129
Average loss at step 48000: 1.655716
Average loss at step 50000: 1.662850
Nearest to 都: 所有人, 个个, 谁, 人人, 不是, 不敢, 众人, 大多,
Nearest to 后: 瞬息, 半晌, 不久, ，, 三年, 虫, 三日, 两三分钟,
Nearest to 着: 盯, 伴随, 抱, 充满, 对, 充斥, 保持, 带,
Nearest to 等: 人, 这, 柳昌, 奇宝, 云韵, 九凤, 阵容, 势力,
Nearest to 与: 九凤, 魂族, 魂玉, 天冥宗, 恐惧, 厌恶, 惊骇, 猎人,
Nearest to 从: 缓缓的, 那, 这里, 其, 其中, 先前, 高空, 远古,
Nearest to 去: 拿, 而, 散, 爆轰而, 赶, 嘴角, 追, ，,
Nearest to 火焰: 四种, 缭绕, 碧绿, 柱, 三种, 紫黑, 深蓝, 熊熊,
Nearest to 一道: 冷喝, 金光, 清脆, 划出, 传出, 光影, 身着, 划起,
Nearest to ”: 闻言, ？, 。, 听得, 么, 彩鳞, 呼, 见到,
Nearest to /: /, :, com, 第百, 第, 第九, 第六, www,
Nearest to 之中: 森林, 眼瞳, 视野, 空间, 同辈, 血池, 遗迹, 阁楼,
Nearest to 一个: 形成, 懒腰, 字, 人, 小辈, 小时, 宛如, 某,
Nearest to 来: 试试, 主持, 袭, 十, 伸出手, 分钟, 找, 攻,
Nearest to 什么: 好处, 做, 意外, 慌, 变故, 事, 不对劲, 东西,
Nearest to 能量: 罩, 体, 九星, 的, 精纯, 浩瀚, 雾气, 光幕,
Average loss at step 52000: 1.645720
Average loss at step 54000: 1.721611
Average loss at step 56000: 1.742322
Average loss at step 58000: 1.645068
Average loss at step 60000: 1.641526
Nearest to 都: 所有人, 大多, ，, 永远, 到处, 能, 竟然, 迟早,
Nearest to 后: 半晌, 不久, 十几分钟, 许久, 完毕, 瞬息, 十分钟, 一番,
Nearest to 着: 瞥, 望, 萧炎望, 散发, 借助, 感受, 把玩, 对,
Nearest to 等: 静, 木辰, 着, 这, 法犸, 云山, 云韵, 你,
Nearest to 与: 法犸, 雅妃, 羡慕, 嘲讽, 云山, 茫然, 大斗师, 奥托,
Nearest to 从: 那, 一旁, 指间, 窗户, 忽然, 刚好, 阳光, 他,
Nearest to 去: 帝都, 撤, 抛, 处行, 。, 先进, 哪里, 找,
Nearest to 火焰: 青白, 中, 罩, 粉红, 三种, 森白, ，, 黑盘,
Nearest to 一道: 光影, 化为, 随着, 留下, 划起, 蕴含着, 影子, 残影,
Nearest to ”: “, 闻言, 啊, 。, ！, ？, !, 么,
Nearest to /: 第三, /, com, 第一千, :, 第六, 第九, 第百,
Nearest to 之中: 经脉, 脑海, 大厅, 气旋, 收进纳, 树林, 黑暗, ，,
Nearest to 一个: 每, 某, 废物, 人情, 形成, 另外, 懒腰, 小,
Nearest to 来: 伸出手, 睁开眼, 十, 袭, 扑, 出身, 惹, 攻,
Nearest to 什么: 做, 事, 不对劲, 有, 东西, 没有, 花招, 造成,
Nearest to 能量: 罩, 柱, 精纯, 冲击波, 壁, 膜, 光幕, 痕迹,
Average loss at step 62000: 1.724995
Average loss at step 64000: 1.658648
Average loss at step 66000: 1.661258
Average loss at step 68000: 1.596577
Average loss at step 70000: 1.635018
Nearest to 都: 大多, 谁, 所有人, 永远, 未曾, 任何人, 无论如何, 知道,
Nearest to 后: 半晌, 背心, 瞬息, ，, 一瞬, 许久, 瞬间, 十几分钟,
Nearest to 着: 望, 拥有, 对, 面对, 伴随, 充斥, 带, 保持,
Nearest to 等: 静, 这, 韩冲, 木辰, 苏媚, 势力, 云督, 人,
Nearest to 与: 洪辰, 沈云, 凶魂, 韩雪, 狂暴, 斗宗, 斗皇, 地位,
Nearest to 从: 天际, 那, 后方, 远处, 天空, 直接, 缓缓的, 沈云,
Nearest to 去: 搽, 而, 看, 赶, 哪里, 嘴角, 散, 追,
Nearest to 火焰: 碧绿, 无形, 罩, 这团, 的, 青白, 寻常, 自,
Nearest to 一道: 化为, 身影, 每, 传来, 血影, 黑线, 黑影, 淡淡的,
Nearest to ”: 闻言, 韩池, 沈云, 洪立, 听得, ？, 苍狼王, “,
Nearest to /: 第九, 第, :, www, 第七, /, 第百一, 第一千,
Nearest to 之中: 脑海, 同辈, 黑暗, 视野, 森林, 营地, 山洞, 房间,
Nearest to 一个: 小时, 不慎, 懒腰, 小辈, 某, 小女孩, 字, 有着,
Nearest to 来: 袭, 追, 出身, 惹, 试试, 打扰, 而, 攻,
Nearest to 什么: 不对劲, 来路, 事, 算不得, 急, 不是, 东西, 可不是,
Nearest to 能量: 天地, 风暴, 狂暴, 涟漪, 罩, 团, 炸声, 匹练,
Average loss at step 72000: 1.534826
Average loss at step 74000: 1.547524
Average loss at step 76000: 1.513826
Average loss at step 78000: 1.509038
Average loss at step 80000: 1.511012
Nearest to 都: 好多年, 大多, ，, 所有人, 明白, 人人, 仿佛, 是因为,
Nearest to 后: ，, 不久, 背心, 虫, 许久, 半月, 半晌, 炼化,
Nearest to 着: 盯, 萧炎望, 充斥, 对, 伴随, 笑, 面对, 还有,
Nearest to 等: 这, 古元, 雷赢, 骨幽, 实力, 萧炎, 人物, 强者,
Nearest to 与: 炎烬, 古元, 雷族, 紫研, 联军, 石族, 萧鼎, 九凤,
Nearest to 从: 其中, 这里, 光, 何处, 缓缓的, 其, 当年, 联军,
Nearest to 去: 而, 抹, 追, 爆轰而, 抓, 赶, ，, 冲,
Nearest to 火焰: 蜥蜴人, 粉红, 滔天, 黑盘, 四种, 晶层, 的, 涌动,
Nearest to 一道: 化为, 流光, 仅仅只是, 身着, 残魂, 黑魔雷, 虹芒, 倩影,
Nearest to ”: ！, 炎烬, 闻言, 雷赢, 魂风, ？, 。, !,
Nearest to /: 第一千, :, 第九, 第, www, /, 第百一, com,
Nearest to 之中: 洞府, 大阵, 山脉, ，, 岩浆, 通道, 眼瞳, 天墓,
Nearest to 一个: 懒腰, 仅仅只是, 不慎, 小时, 势力, 魂族, 不小, 是,
Nearest to 来: 涌, 攻, 转过身, 而, 蹭, 主持, 派, ，,
Nearest to 什么: 叫做, 做, 东西, 有着, 容易, 不成, 没有, 有,
Nearest to 能量: 核, 天地, 涟漪, 风暴, 狂暴, 的, 精纯, 浩瀚,
Average loss at step 82000: 1.644444
Average loss at step 84000: 1.548087
Average loss at step 86000: 1.529392
Average loss at step 88000: 1.584355
Average loss at step 90000: 1.569740
Nearest to 都: 始终, 每天, 所有人, 是因为, 无论如何, 大多, 好多年, ，,
Nearest to 后: 片刻, 许久, 半晌, 不久, ，, 十分钟, 两天, 瞬间,
Nearest to 着: 带, 笑, 对, 面对, 借助, 仗, 咬, 冒,
Nearest to 等: 苏笑, 人, 阿泰, 静, 修崖, 这, 奇宝, 严皓,
Nearest to 与: 白程, 修岩, 琥嘉, 姚盛, 斗灵, 杀意, 吴昊, 大斗师,
Nearest to 从: 指间, 脚底, 窗户, 那里, 阳光, 至, 各处, 塔中,
Nearest to 去: 抛, 抹, 跑, 投, 挑战, 。, 哪, 走,
Nearest to 火焰: 柱, 森白, 蜥蜴人, 尖刺, ，, 青白, 盔甲, 的,
Nearest to 一道: 残影, 人影, 苍老, 闷响, 白影, 发出, 划出, 细小,
Nearest to ”: ！, 听得, 高手, 姚盛, 闻言, 强榜, 紫研, 。,
Nearest to /: 第, :, /, com, 第一千, 第百, 第百一, www,
Nearest to 之中: 木盆, 脑海, 眼瞳, 收进纳, 经脉, 场地, 大厅, 这内院,
Nearest to 一个: 小时, 每, 冷颤, 团队, 小女孩, 回合, 将近, 玉瓶,
Nearest to 来: 出身, 转过身, 伸出手, 日, 找麻烦, 扑, 分辨, 攻,
Nearest to 什么: 算不得, 急, 变故, 发生, 谈不上, 差错, 办法, 好,
Nearest to 能量: 壁, 炸响, 团, 膜, 球, 柱, 涟漪, 残剑,
Average loss at step 92000: 1.557906
Average loss at step 94000: 1.512269
Average loss at step 96000: 1.509176
Average loss at step 98000: 1.478617
Average loss at step 100000: 1.457082
Nearest to 都: 大多, 所有人, 人人, 清楚, 谁, 每天, 一辈子, 始终,
Nearest to 后: 片刻, 一会, 半晌, 许久, 背心, ，, 瞬息, 出现,
Nearest to 着: 望, 对, 盯, 充斥, 带, 噙, 面对, 笑,
Nearest to 等: 静, 妖女, 这, 萧炎, 宋清, 宝贝, 天地, 凶悍,
Nearest to 与: 曹家, 欣蓝, 宋清, 丹晨, 激动, 萧炎, 曹颖, 叶家,
Nearest to 从: 光, 外面, 想要, 天空, 其内, 灵魂深处, 其, 叶家,
Nearest to 去: 散, 而, 投, 丹塔, 抓, 冲, 。, 叶家,
Nearest to 火焰: 蜥蜴人, 柱, 罩, 缭绕, 无形, 三种, 风暴, 的,
Nearest to 一道: 发出, 细微, 黑影, 苍老, 划起, 血影, 低沉, 化为,
Nearest to ”: ！, ？, 听得, 说完, 紫研, 见到, 。, 语罢,
Nearest to /: 第七, 第九, /, :, www, 第一千, 第, 第百一,
Nearest to 之中: 视野, 森林, 院落, 同辈, 木盆, 眼瞳, 火焰, 视线,
Nearest to 一个: 另外, 每, 某, 不慎, 将近, 有着, 方向, 冷颤,
Nearest to 来: 十, 袭, 找麻烦, 而, 攻, 试试, 敢, 惹,
Nearest to 什么: 意外, 东西, 岔子, 算不得, 差错, 说, 有, 伤害,
Nearest to 能量: 天地, 风暴, 膜, 罩, 涟漪, 漩涡, 的, 青红,

Process finished with exit code 0

```