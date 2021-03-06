# efficient-pinyin-keys

高效拼音输入键位

---
# 目录

- [efficient-pinyin-keys](#efficient-pinyin-keys)
- [目录](#目录)
- [Step 1 统计按键使用频率](#step-1-统计按键使用频率)
  - [热度图](#热度图)
  - [按键频率分布直方图](#按键频率分布直方图)
- [Step 2 均衡性 & 输入效率](#step-2-均衡性--输入效率)
  - [均衡性](#均衡性)
  - [输入效率](#输入效率)
- [Step 3 改进编码方案](#step-3-改进编码方案)
  - [分析与改进](#分析与改进)
    - [分析](#分析)
    - [改进](#改进)
  - [改进方案](#改进方案)
    - [全拼改进方案](#全拼改进方案)
    - [小鹤双拼方案](#小鹤双拼方案)
    - [小鹤双拼改进方案](#小鹤双拼改进方案)
  - [测试结果分析](#测试结果分析)
- [参考资料](#参考资料)

---


# Step 1 统计按键使用频率

- 要求: 给定一篇中文文章(input.txt), 统计出使用全拼输入法录入这篇文章时26个字母键的使用频率, 绘制热力图。
- 输入: 一篇中文文章（附件文章）
- 输出: 录入这篇文章的26字母键使用热力图

## 热度图

- [全拼热度图](https://zuoqiu-yingyi.github.io/efficient-pinyin-keys/src/www/full.html)
- [声母 & (`y`,`w`)热度图](https://zuoqiu-yingyi.github.io/efficient-pinyin-keys/src/www/initials.html)
- [韵母 & 介母热度图](https://zuoqiu-yingyi.github.io/efficient-pinyin-keys/src/www/finals.html)

## 按键频率分布直方图
![频率分布直方图](./src/images/频率分布直方图.png)


# Step 2 均衡性 & 输入效率

- 要求: 设计评价标准来分别评价使用全拼录入这篇文章时的按键使用均衡性和输入效率（请根据个人理解自行定义，建议使用明确的量化指标）。 
- 输出: 量化评价标准或方法，以及对全拼输入方案的评价结果

## 均衡性

`均衡性`可以由`按键熵`, `手指熵`与`手掌熵`三个指标的加权和表示:
$$
B = A_k \cdot B_k \cdot W_k + A_f \cdot B_f \cdot W_f + A_h \cdot B_h \cdot W_h
$$
- $B$: `均衡性`
- $B_k$: `按键熵`, 使用`各按键实际敲击频率`的熵表示, 衡量按键均匀敲击的程度
    - $B_k \to 0$: 按键实际使用概率分布不均匀
    - $B_k \to 1$: 按键实际使用概率分布均匀
- $B_f$: `手指熵`, 使用`各手指实际敲击频率`的熵表示, 衡量手指均匀敲击的程度
    - $B_f \to 0$: 手指实际使用概率分布不均匀
    - $B_f \to 1$: 手指实际使用概率分布均匀
- $B_h$: `手掌熵`, 使用`各手掌实际使用频率`的熵表示, 衡量手掌均匀敲击的程度
    - $B_h \to 0$: 手掌实际使用概率分布不均匀
    - $B_h \to 1$: 手掌实际使用概率分布均匀
- $A_k$: `按键频率熵归一化系数`, 值为`按键频率熵`最大熵的倒数 $log_2{26}$
- $A_f$: `手指频率熵归一化系数`, 值为`手指频率熵`最大熵的倒数 $log_2{8}$
- $A_h$: `手掌频率熵归一化系数`, 值为`手掌频率熵`最大熵的倒数 $log_2{2}$
- $W_k$: `按键均匀性权重`
- $W_f$: `手指均匀性权重`
- $W_h$: `手掌均匀性权重`

`熵`:
$$
H(p) := -\sum_i{p(i) \log_2{p(i)}}
$$

`交叉熵`:
$$
H(p \parallel q) := - \sum_i{p(i) \log_2{q(i)}}
$$


## 输入效率

`输入效率`可分别由`平均单字击键数量`与`平均单字消耗时间`表征

1. **平均单字击键数:**
$$
\bar{N} := \cfrac{N_k}{N_c}
$$
- $\bar{N}$: `平均单字击键数`
- $N_k$: `样本文章击键总数`
- $N_c$: `样本文章总字数`(不含标点符号)

2. **平均单字耗时:**
$$
\bar{T} := \cfrac{\sum_i^{N_k}{F_i \cdot R_i \cdot C_i \cdot H_i \cdot T_0}}{N_c}
$$
- $\bar{T}$: `平均单字耗时`
- $F_i$: `手指修正参数`, 表征不同手指敲击所用时间差异(食指 < 中指 < 无名指 < 小指)
- $R_i$: `键位修正参数(行)`, 表征不同行键位敲击所用时间差异(第2行 < 第1行 < 第3行)
- $C_i$: `键位修正参数(列)`, 表征不同列键位敲击所用时间差异(其他 < `TGB` = `YHN`)
- $H_i$: `手掌修正参数`, 两手交替击键与单手顺次击键所用时间差异(前后两次击键使用不同手 < 前后两次击键使用相同手)
- $T_0$: `单键敲击基本耗时`
- $N_k$: `样本文章击键总数`
- $N_c$: `样本文章总字数`(不含标点符号)


# Step 3 改进编码方案

- 要求: 基于在`Step 2`中制定的标准，尝试在全拼基础上改进打字编码方案，使得输入该文章时字母键的使用更加均衡、输入更加高效，展示改进的结果并分析。
- 输入: 一篇中文文章（附件文章）
- 输出: 新的打字编码方案、新旧方案在均衡性和输入效率方面的对比

## 分析与改进

### 分析

1. 对比`全拼按键频率分布直方图`, `声母 & (y,w)频率分布直方图` 与 `韵母 & 介母频率分布直方图`, 可以看出相对于全拼与声母, 韵母数量更多且更加集中(例如<kbd>i</kbd>和<kbd>e</kbd>)

2. 根据使用`平均单字消耗时间`作为输入效率量化方案, 可以计算出各按键的权重(`如下表所示`), 其中第一行中的<kbd>E</kbd>, <kbd>R</kbd>, <kbd>T</kbd>, <kbd>Y</kbd>, <kbd>U</kbd>, <kbd>I</kbd>, 第二行中的<kbd>S</kbd>, <kbd>D</kbd>, <kbd>F</kbd>, <kbd>G</kbd>, <kbd>H</kbd>, <kbd>J</kbd>, <kbd>E</kbd>, <kbd>K</kbd>, <kbd>L</kbd>, 第三行中的<kbd>V</kbd>, <kbd>M</kbd>等键均具有较好的输入效率

| 行号  |        第1列         |        第2列         |        第3列         |        第4列         |        第5列         |        第6列         |        第7列         |        第8列         |        第9列         |        第10列        |
| :---: | :------------------: | :------------------: | :------------------: | :------------------: | :------------------: | :------------------: | :------------------: | :------------------: | :------------------: | :------------------: |
| 第1行 | <kbd>Q</kbd> 1.68750 | <kbd>W</kbd> 1.40625 | <kbd>E</kbd> 1.26562 | <kbd>R</kbd> 1.12500 | <kbd>T</kbd> 1.26562 | <kbd>Y</kbd> 1.26562 | <kbd>U</kbd> 1.12500 | <kbd>I</kbd> 1.26562 | <kbd>O</kbd> 1.40625 | <kbd>P</kbd> 1.68750 |
| 第2行 | <kbd>A</kbd> 1.50000 | <kbd>S</kbd> 1.25000 | <kbd>D</kbd> 1.12500 | <kbd>F</kbd> 1.00000 | <kbd>G</kbd> 1.12500 | <kbd>H</kbd> 1.12500 | <kbd>J</kbd> 1.00000 | <kbd>K</kbd> 1.12500 | <kbd>L</kbd> 1.25000 |                      |
| 第3行 | <kbd>Z</kbd> 1.87500 | <kbd>X</kbd> 1.56250 | <kbd>C</kbd> 1.40625 | <kbd>V</kbd> 1.25000 | <kbd>B</kbd> 1.40625 | <kbd>N</kbd> 1.40625 | <kbd>M</kbd> 1.25000 |                      |                      |                      |

3. 分析现有双拼方案(以下图所示`小鹤双拼`为例)
    1. 部分韵母(介母)由于其拼写规则的互斥(如下所示), 合并后并不会使重码率出现上升
        - `üe` & `ue`
        - `o` & `uo`
        - `ong` & `iong`
        - `ing` & `uai`
        - `iang` & `uang`
        - `ia` & `ua`
        - `ü` & `ui`
    2. 特殊音节可使用两个互斥的按键表示(如下所示)
        - `a`: <kbd>A</kbd> <kbd>A</kbd>
        - `ai`: <kbd>A</kbd> <kbd>I</kbd>
        - `an`: <kbd>A</kbd> <kbd>N</kbd>
        - `ang`: <kbd>A</kbd> <kbd>H</kbd>
        - `ao`: <kbd>A</kbd> <kbd>O</kbd>
        - `e`: <kbd>E</kbd> <kbd>E</kbd>
        - `ei`: <kbd>E</kbd> <kbd>I</kbd>
        - `en`: <kbd>E</kbd> <kbd>N</kbd>
        - `er`: <kbd>E</kbd> <kbd>R</kbd>
        - `eng`: <kbd>E</kbd> <kbd>G</kbd>
        - `o`: <kbd>O</kbd> <kbd>O</kbd>
        - `ou`: <kbd>O</kbd> <kbd>U</kbd>
    3. 为了降低学习门槛, 该双拼方案保持大多数声母与单韵母的位置不变, 因此其击键热度图热区并非完全集中于效率最高的键位

![小鹤双拼](./src/images/小鹤双拼.png)
<center>小鹤双拼布局与击键热度图</center>


### 改进

由以上分析结果, 针对全拼与双拼, 先提出两种改进方案
1. `全拼改进方案`: 综合`全拼按键频率分布直方图`与`按键输入效率权重表`, 将高频字母映射至输入效率较高的按键, 以期在保持`按键熵` $B_k$ 不变的条件下提高`手指熵` $B_f$, 进而提高其`均衡性` $B$, 同时提高其使用`平均单字耗时` $\bar{T}$ 表征的`输入效率`
2. `双拼改进方案`: 综合`声母 & (y,w)频率分布直方图`, `韵母 & 介母频率分布直方图`与`小鹤双拼布局与击键热度图`, 在保证不发生按键冲突的前提下, 将高频声母与韵母映射至输入效率较高的按键, 以期同时提高`均衡性` $B$, `平均单字击键数` $\bar{N}$ 与 `平均单字耗时` $\bar{T}$


## 改进方案


### 全拼改进方案

- [改进方案热度图](https://zuoqiu-yingyi.github.io/efficient-pinyin-keys/src/www/ifull.html)

### 小鹤双拼方案

- [改进方案热度图](https://zuoqiu-yingyi.github.io/efficient-pinyin-keys/src/www/xiaohe.html)

### 小鹤双拼改进方案

- [改进方案热度图](https://zuoqiu-yingyi.github.io/efficient-pinyin-keys/src/www/ixiaohe.html)


## 测试结果分析

|    方案    |  均衡性  |  按键熵  |  手指熵  |  手掌熵  | 平均单字击键数 | 击键总数 | 样本字数 | 平均单字耗时 |    总耗时    |
| :--------: | :------: | :------: | :------: | :------: | :------------: | :------: | :------: | :----------: | :----------: |
| 全拼改进前 | 0.897946 | 4.067694 | 2.597974 | 0.995023 |    2.952491    |  53508   |  18123   |   3.263177   | 59138.558594 |
| 全拼改进后 | 0.877743 | 4.067694 | 2.361951 | 0.992884 |    2.952491    |  53508   |  18123   |   3.000504   | 54378.140625 |
| 双拼改进前 | 0.944259 | 4.420238 | 2.694713 | 0.998021 |    2.000000    |  36246   |  18123   |   2.240133   | 40597.937500 |
| 双拼改进后 | 0.932026 | 4.420238 | 2.584392 | 0.985862 |    2.000000    |  36246   |  18123   |   2.108291   | 38208.562500 |

1. 对比全拼改进前后与双拼改进前后的测试结果:
    - 通过更改键盘按键布局对`平均单字耗时`衡量的输入效率具有一定的提升, 但是不影响`平均单字击键数`
    - 通过更改键盘按键布局对方案的`按键熵`无影响, 对`手掌熵`影响有限, 而`手指熵`会显著降低, 进而导致`均衡性`也随之降低, 针对对同一种方案改进前后, `均衡性`与`输入效率`可能存在一定的负相关, 即按键击键越均衡输入效率反而会下降
2. 对比全拼与双拼的测试结果:
    - 双拼方案可以有效提高`均衡性`与`输入效率`

# 参考资料

1. [Keyboard Heatmap | Realtime heatmap visualization of text character distribution](https://www.patrick-wied.at/projects/heatmap-keyboard/)
0. [双拼布局的统计计算](https://tiansh.github.io/lqbz/sp/)
0. [GitHub - tiansh/lqbz: 乱七八糟——堆一些奇怪的东西](https://github.com/tiansh/lqbz)
0. [heatmap.js : Dynamic Heatmaps for the Web](https://www.patrick-wied.at/static/heatmapjs/)
0. [GitHub - pa7/heatmap.js: 🔥 JavaScript Library for HTML5 canvas based heatmaps](https://github.com/pa7/heatmap.js)
0. [python使用matplotlib画词频分析的图_如何使用matplotlib或任何其他库创建词频图_weixin_39746652的博客-CSDN博客](https://blog.csdn.net/weixin_39746652/article/details/113969749)
0. [python画频率直方图_字母频率：绘制一个直方图，按PYTHON的值排序_weixin_39526459的博客-CSDN博客](https://blog.csdn.net/weixin_39526459/article/details/109970976)
0. [Matplotlib中柱状图bar使用 - 1直在路上1 - 博客园](https://www.cnblogs.com/always-fight/p/9707727.html)
0. [numpy - Fastest way to compute entropy in Python - Stack Overflow](https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python)
0. [API — pypinyin 0.42.0 文档](https://pypinyin.readthedocs.io/zh_CN/master/api.html)
0. [LaTeX公式手册(全网最全) - 樱花赞 - 博客园](https://www.cnblogs.com/1024th/p/11623258.html)
0. [介母（拼音符号及组合特点）_百度百科](https://baike.baidu.com/item/%E4%BB%8B%E6%AF%8D/4028328)
0. [汉语拼音方案 - 中华人民共和国教育部政府门户网站](http://www.moe.gov.cn/jyb_sjzl/ziliao/A19/195802/t19580201_186000.html)
0. [python-pinyin/test_standard.py at master · mozillazg/python-pinyin](https://github.com/mozillazg/python-pinyin/blob/master/tests/test_standard.py)
0. [小鹤双拼·小鹤音形 - 官方网站](https://www.flypy.com/)
0. [Python格式化字符串 - 田小计划 - 博客园](https://www.cnblogs.com/wilber2013/p/4641616.html)

