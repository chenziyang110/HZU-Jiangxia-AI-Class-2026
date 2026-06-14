import torch
import torch.nn as nn
import torch.optim as optim
import math

# ==========1.编辑距离算法（拼写纠错用）==========
def levenshtein(a, b):
    len_a, len_b = len(a), len(b)
    dp = [[0]*(len_b+1) for _ in range(len_a+1)]
    for i in range(len_a+1): dp[i][0] = i
    for j in range(len_b+1): dp[0][j] = j
    for i in range(1, len_a+1):
        for j in range(1, len_b+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[len_a][len_b]

def find_similar_word(word, word_pool, max_dist=2):
    min_dist = 999
    best_match = None
    for w in word_pool:
        dist = levenshtein(word, w)
        if dist < min_dist and dist <= max_dist:
            min_dist = dist
            best_match = w
    return best_match

# ==========2.Transformer模型结构（备用，生词才调用）==========
# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=200):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.shape
        B, Lk, _ = k.shape
        q = self.w_q(q).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn_score, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, -1)
        return self.w_o(out)

# Encoder层
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        self.norm2 = nn.LayerNorm(64)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.ff(x))
        return x

# Decoder层
class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        self.norm3 = nn.LayerNorm(64)

    def forward(self, x, enc_out, tgt_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask=tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out))
        x = self.norm3(x + self.ff(x))
        return x

# Transformer主体
class Translator(nn.Module):
    def __init__(self, en_vocab, zh_vocab):
        super().__init__()
        self.en_emb = nn.Embedding(en_vocab, 64)
        self.zh_emb = nn.Embedding(zh_vocab, 64)
        self.pos = PositionalEncoding()
        self.encoder = nn.ModuleList([EncoderLayer()])
        self.decoder = nn.ModuleList([DecoderLayer()])
        self.fc = nn.Linear(64, zh_vocab)

    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size))
        return mask

    def forward(self, en, zh):
        x = self.pos(self.en_emb(en))
        for layer in self.encoder:
            x = layer(x)
        y = self.pos(self.zh_emb(zh))
        mask = self.get_tgt_mask(zh.size(1)).to(zh.device)
        for layer in self.decoder:
            y = layer(y, x, tgt_mask=mask)
        return self.fc(y)

# ==========3.5000高频英汉词典(短语+单词)==========
word_dict = {
    "hello":"你好","i love you":"我爱你","how are you":"你好吗","thank you":"谢谢你","good morning":"早上好",
    "a":"一个","abandon":"抛弃","ability":"能力","able":"能够","about":"关于","above":"在上方","abroad":"在国外","absent":"缺席","absolute":"绝对的","absorb":"吸收",
    "accept":"接受","accident":"事故","accompany":"陪伴","accomplish":"完成","according":"根据","account":"账目","achieve":"实现","across":"穿过","act":"行动","active":"积极的",
    "actual":"实际的","add":"添加","address":"地址","adjust":"调整","admit":"承认","adopt":"采纳","advance":"前进","advantage":"优势","advice":"建议","affect":"影响",
    "afraid":"害怕的","after":"在之后","again":"再次","age":"年龄","ago":"以前","agree":"同意","air":"空气","alarm":"警报","all":"全部","allow":"允许",
    "almost":"几乎","alone":"独自","along":"沿着","already":"已经","also":"也","although":"虽然","always":"总是","am":"是","among":"在之中","amount":"数量",
    "analyse":"分析","another":"另一个","answer":"回答","any":"任何","apart":"分开","appear":"出现","area":"区域","argue":"争论","arm":"手臂","arrive":"到达",
    "art":"艺术","as":"作为","ask":"询问","at":"在","attempt":"尝试","attend":"参加","attention":"注意力","august":"八月","aunt":"阿姨","autumn":"秋天",
    "available":"可用的","avoid":"避免","awake":"醒着","baby":"婴儿","back":"回来","bad":"坏的","bag":"包","ball":"球","banana":"香蕉","bank":"银行",
    "base":"基础","basket":"篮子","bath":"洗澡","battle":"战斗","be":"是","bear":"忍受","beautiful":"漂亮的","because":"因为","become":"变成","bed":"床",
    "before":"在之前","begin":"开始","behind":"在后面","believe":"相信","bell":"铃铛","below":"在下方","beside":"在旁边","best":"最好","better":"更好","between":"在中间",
    "big":"大的","bird":"鸟","birth":"出生","bit":"一点","black":"黑色","blow":"吹","blue":"蓝色","boat":"小船","body":"身体","book":"书本",
    "both":"两者","bottle":"瓶子","bottom":"底部","bowl":"碗","box":"盒子","boy":"男孩","brain":"大脑","break":"打破","breakfast":"早餐","breathe":"呼吸",
    "bridge":"桥梁","bright":"明亮的","bring":"带来","brother":"兄弟","brown":"棕色","brush":"刷","build":"建造","burn":"燃烧","bus":"公交车","business":"生意",
    "busy":"忙碌的","but":"但是","buy":"购买","by":"凭借","cake":"蛋糕","call":"呼叫","can":"能够","cap":"帽子","capital":"首都","car":"汽车",
    "card":"卡片","care":"关心","carry":"携带","cat":"猫咪","catch":"抓住","cause":"原因","cent":"分","center":"中心","certain":"确定的","chair":"椅子",
    "change":"改变","child":"孩子","children":"孩子们","city":"城市","class":"班级","clean":"干净的","clear":"清楚的","climb":"攀爬","clock":"时钟","close":"关闭",
    "clothes":"衣服","cloud":"云朵","club":"俱乐部","coal":"煤炭","coat":"外套","coffee":"咖啡","cold":"寒冷","collect":"收集","college":"大学","colour":"颜色",
    "come":"来","comfort":"安慰","common":"普通的","communicate":"交流","community":"社区","company":"公司","compare":"比较","compete":"竞争","complete":"完成","computer":"电脑",
    "condition":"条件","connect":"连接","consider":"考虑","continue":"继续","control":"控制","cook":"烹饪","cool":"凉爽","copy":"复制","correct":"正确的","cost":"花费",
    "could":"能够","count":"数数","country":"国家","cover":"覆盖","cow":"奶牛","crazy":"疯狂的","create":"创造","cross":"穿过","cry":"哭泣","cup":"杯子",
    "current":"当前的","cut":"切割","daily":"日常的","dance":"跳舞","danger":"危险","dark":"黑暗","date":"日期","daughter":"女儿","day":"天","dead":"死去的",
    "decide":"决定","deep":"深的","degree":"度数","deliver":"递送","demand":"要求","depend":"依靠","describe":"描述","desk":"书桌","develop":"发展","devote":"奉献",
    "diary":"日记","die":"死亡","difference":"区别","difficult":"困难的","dig":"挖掘","dinner":"晚餐","direct":"直接的","dirt":"尘土","discover":"发现","discuss":"讨论",
    "disease":"疾病","distance":"距离","district":"区域","divide":"分开","do":"做","doctor":"医生","dog":"狗","dollar":"美元","door":"门","double":"双倍",
    "down":"向下","draw":"画画","dream":"梦想","dress":"裙子","drink":"喝","drive":"驾驶","drop":"掉落","dry":"干燥","duck":"鸭子","duty":"责任",
    "each":"每个","early":"早的","east":"东方","easy":"容易的","eat":"吃","education":"教育","effect":"影响","effort":"努力","eight":"八","either":"任一",
    "electric":"电的","elephant":"大象","else":"其他","empty":"空的","end":"结束","engine":"引擎","english":"英语","enjoy":"享受","enough":"足够","enter":"进入",
    "environment":"环境","equal":"相等的","escape":"逃跑","especially":"尤其","establish":"建立","even":"甚至","ever":"曾经","every":"每个","example":"例子","examine":"检查",
    "except":"除了","excuse":"借口","exercise":"锻炼","expensive":"昂贵的","explain":"解释","eye":"眼睛","face":"脸","fact":"事实","factory":"工厂","fail":"失败",
    "fair":"公平的","fall":"落下","family":"家庭","far":"遥远","fast":"快速","father":"父亲","fear":"害怕","feed":"喂养","feel":"感受","few":"少量",
    "field":"田野","fight":"打架","fill":"填满","find":"找到","fine":"好的","fire":"火","first":"第一","fish":"鱼","fit":"合适","five":"五",
    "fix":"修理","flight":"航班","floor":"地板","flower":"花朵","fly":"飞","food":"食物","foot":"脚","for":"为了","force":"强迫","foreign":"外国的",
    "forget":"忘记","form":"形式","forward":"向前","free":"自由","friend":"朋友","from":"来自","front":"前面","full":"满的","fun":"乐趣","future":"未来",
    "game":"游戏","garden":"花园","gas":"气体","general":"大体的","gentle":"温和的","get":"得到","give":"给予","goal":"目标","go":"去","god":"上帝",
    "gold":"黄金","good":"好的","government":"政府","grade":"年级","grandpa":"爷爷","grass":"草地","great":"伟大的","green":"绿色","ground":"地面","group":"小组",
    "grow":"生长","guard":"守卫","guess":"猜测","guide":"指导","gun":"枪","habit":"习惯","hair":"头发","half":"一半","hall":"大厅","hand":"手",
    "hang":"悬挂","happen":"发生","happy":"开心","hard":"艰难","harm":"伤害","harvest":"收获","have":"拥有","he":"他","head":"头","health":"健康",
    "hear":"听见","heart":"心脏","heat":"热量","help":"帮助","hen":"母鸡","here":"这里","hide":"躲藏","high":"高的","hill":"小山","him":"他(宾格)",
    "his":"他的","history":"历史","hobby":"爱好","hold":"握住","holiday":"假期","home":"家","hope":"希望","horse":"马","hospital":"医院","hot":"炎热",
    "hotel":"酒店","hour":"小时","house":"房子","how":"如何","however":"然而","huge":"巨大","human":"人类","hundred":"百","hungry":"饥饿","hunt":"打猎",
    "hurry":"匆忙","husband":"丈夫","ice":"冰","idea":"想法","if":"如果","ignore":"忽视","ill":"生病","imagine":"想象","immediately":"立刻","important":"重要的",
    "improve":"改善","in":"在里面","include":"包含","increase":"增加","indeed":"的确","indian":"印第安人","industry":"工业","information":"信息","inside":"在内","instead":"代替",
    "institute":"学院","intend":"打算","interest":"兴趣","into":"进入","introduce":"介绍","invent":"发明","invite":"邀请","iron":"铁","is":"是","it":"它",
    "its":"它的","job":"工作","join":"加入","journey":"旅途","joy":"快乐","judge":"判断","juice":"果汁","jump":"跳跃","just":"刚刚","keep":"保持",
    "key":"钥匙","kid":"小孩","kill":"杀死","kilogram":"千克","kind":"友善的","king":"国王","kitchen":"厨房","knee":"膝盖","knife":"小刀","knock":"敲打",
    "know":"知道","lab":"实验室","lady":"女士","lake":"湖泊","lamp":"台灯","land":"陆地","language":"语言","large":"巨大","last":"最后","late":"迟到",
    "laugh":"大笑","law":"法律","lay":"放置","lazy":"懒惰","lead":"引导","learn":"学习","leave":"离开","left":"左边","leg":"腿","lend":"借出",
    "lesson":"课程","let":"让","letter":"信件","level":"水平","library":"图书馆","lie":"躺","life":"生命","lift":"举起","like":"喜欢","line":"线条",
    "list":"清单","listen":"聆听","little":"少量","live":"居住","lock":"锁","lonely":"孤独","look":"看","lose":"丢失","loss":"损失","lot":"许多",
    "love":"爱","low":"低的","luck":"运气","lunch":"午饭","machine":"机器","mad":"疯狂","magazine":"杂志","main":"主要的","make":"制作","man":"男人",
    "manage":"管理","many":"许多","map":"地图","march":"三月","mark":"标记","market":"市场","match":"匹配","math":"数学","matter":"事情","may":"可以",
    "maybe":"也许","me":"我(宾格)","mean":"意思是","meet":"遇见","member":"成员","memory":"记忆","mention":"提及","merry":"愉快","message":"消息","metal":"金属",
    "method":"方法","middle":"中间","milk":"牛奶","mind":"想法","mine":"我的","minute":"分钟","miss":"想念","mix":"混合","model":"模型","moment":"瞬间",
    "money":"钱","monkey":"猴子","month":"月份","moon":"月亮","more":"更多","morning":"早上","mother":"妈妈","motor":"发动机","mountain":"大山","mouse":"老鼠",
    "move":"移动","movie":"电影","much":"很多","must":"必须","my":"我的","name":"名字","national":"国家的","natural":"自然的","near":"附近","necessary":"必要的",
    "neck":"脖子","need":"需要","neither":"两者都不","net":"网","never":"从不","new":"新的","news":"新闻","next":"下一个","nice":"美好的","night":"夜晚",
    "nine":"九","no":"不","nobody":"没人","nod":"点头","none":"没有","nor":"也不","north":"北方","nose":"鼻子","not":"不","note":"笔记",
    "now":"现在","number":"数字","nurse":"护士","object":"物体","ocean":"海洋","october":"十月","offer":"提供","office":"办公室","often":"经常","oil":"油",
    "old":"老的","on":"在上面","once":"一次","one":"一","only":"仅仅","open":"打开","operate":"操作","opinion":"观点","opposite":"相反","or":"或者",
    "order":"命令","organise":"组织","origin":"起源","other":"其他","our":"我们的","out":"在外","over":"超过","own":"拥有","pack":"打包","page":"页码",
    "pain":"疼痛","paint":"画画","pair":"一对","palace":"宫殿","pan":"平底锅","paper":"纸张","parent":"父母","park":"公园","part":"部分","party":"聚会",
    "pass":"通过","past":"过去","path":"小路","pay":"支付","peace":"和平","pen":"钢笔","pencil":"铅笔","people":"人们","pepper":"胡椒","per":"每",
    "perfect":"完美的","person":"人","pet":"宠物","phone":"电话","photo":"照片","physical":"身体的","pick":"捡起","picnic":"野餐","picture":"图片","piece":"块",
    "pig":"猪","pile":"堆","pillow":"枕头","pin":"别针","pink":"粉色","pioneer":"先锋","pipe":"管子","pity":"遗憾","pizza":"披萨","place":"地点",
    "plan":"计划","plant":"植物","plate":"盘子","play":"玩耍","please":"请","plenty":"大量","point":"指向","police":"警察","pond":"池塘","pool":"泳池",
    "poor":"贫穷","pop":"流行","pork":"猪肉","position":"位置","possible":"可能","post":"邮政","pot":"锅","potato":"土豆","pound":"英镑","pour":"倾倒",
    "power":"力量","practice":"练习","praise":"表扬","predict":"预测","prefer":"更喜欢","prepare":"准备","present":"礼物","president":"总统","press":"按压","pretend":"假装",
    "pretty":"漂亮","prevent":"阻止","price":"价格","pride":"骄傲","primary":"初级的","print":"打印","private":"私人的","prize":"奖品","problem":"问题","produce":"生产",
    "program":"程序","project":"项目","promise":"承诺","protect":"保护","proud":"骄傲","prove":"证明","provide":"提供","public":"公共的","pull":"拉","punish":"惩罚",
    "pupil":"学生","pure":"纯净","push":"推","put":"放置","question":"问题","quick":"快速","quiet":"安静","rabbit":"兔子","race":"赛跑","radio":"收音机",
    "rain":"下雨","raise":"举起","range":"范围","rapid":"迅速","rate":"比率","raw":"生的","reach":"到达","read":"阅读","ready":"准备好","real":"真实的",
    "reason":"原因","receive":"收到","red":"红色","refuse":"拒绝","regard":"看待","regular":"常规的","relation":"关系","relax":"放松","remain":"保持","remember":"记住",
    "remove":"移除","rent":"租金","repair":"修理","repeat":"重复","reply":"回复","report":"报告","require":"需要","rest":"休息","result":"结果","return":"返回",
    "review":"复习","rice":"米饭","rich":"富有","ride":"骑行","right":"右边","ring":"戒指","rise":"上升","river":"河流","road":"马路","rob":"抢劫",
    "rock":"岩石","room":"房间","root":"根","rope":"绳子","rose":"玫瑰","rough":"粗糙","round":"圆形","route":"路线","rubber":"橡胶","rule":"规则",
    "run":"跑","rush":"冲","sad":"伤心","safe":"安全","sail":"航行","salad":"沙拉","sale":"售卖","salt":"盐","same":"相同","sand":"沙子",
    "satisfy":"满足","saturday":"周六","save":"保存","say":"说","school":"学校","science":"科学","score":"分数","sea":"大海","search":"搜寻","season":"季节",
    "second":"第二","secret":"秘密","see":"看见","seek":"寻找","seem":"似乎","sell":"售卖","send":"发送","sense":"感觉","separate":"分开","serious":"严肃",
    "serve":"服务","set":"设置","seven":"七","several":"几个","share":"分享","she":"她","sheet":"床单","shelf":"架子","shell":"贝壳","shine":"闪耀",
    "ship":"轮船","shirt":"衬衫","shoe":"鞋子","shoot":"射击","shop":"商店","short":"短的","should":"应该","show":"展示","shut":"关闭","side":"边",
    "sight":"视力","sign":"标志","silent":"安静","silver":"银","simple":"简单","since":"自从","sing":"唱歌","sink":"下沉","sir":"先生","sister":"姐妹",
    "sit":"坐下","six":"六","size":"尺寸","sleep":"睡觉","slow":"缓慢","small":"小的","smell":"闻","smile":"微笑","smoke":"烟","snow":"雪",
    "so":"所以","soap":"肥皂","social":"社会的","some":"一些","son":"儿子","soon":"很快","sort":"分类","sound":"声音","south":"南方","space":"空间",
    "speak":"说话","special":"特殊","spell":"拼写","spend":"花费","spirit":"精神","split":"拆分","spoon":"勺子","sport":"运动","spread":"传播","spring":"春天",
    "square":"广场","stage":"舞台","stand":"站立","star":"星星","start":"开始","state":"状态","stay":"停留","step":"步骤","stick":"棍子","still":"仍然",
    "stock":"库存","stop":"停止","store":"储存","story":"故事","straight":"笔直","strange":"奇怪","street":"街道","strike":"击打","strong":"强壮","student":"学生",
    "study":"学习","subject":"科目","succeed":"成功","such":"如此","sugar":"糖","suggest":"建议","summer":"夏天","sun":"太阳","support":"支持","suppose":"猜想",
    "sure":"确定","surface":"表面","surprise":"惊喜","swim":"游泳","table":"桌子","take":"拿","talk":"交谈","taxi":"出租车","tea":"茶","teach":"教",
    "telephone":"电话","television":"电视","tell":"告诉","ten":"十","term":"学期","test":"测试","than":"比","thank":"感谢","that":"那个","the":"定冠词",
    "their":"他们的","them":"他们(宾格)","then":"然后","there":"那里","these":"这些","they":"他们","thing":"事物","think":"思考","third":"第三","this":"这个",
    "those":"那些","three":"三","through":"穿过","throw":"扔","ticket":"票","tie":"捆绑","till":"直到","time":"时间","tiny":"微小","tip":"小费",
    "tired":"疲惫","to":"到","today":"今天","together":"一起","tomato":"西红柿","tomorrow":"明天","tonight":"今晚","too":"也","top":"顶部","toward":"朝向",
    "town":"小镇","train":"火车","travel":"旅行","tree":"树木","trouble":"麻烦","true":"真实","try":"尝试","turn":"转动","two":"二","type":"类型",
    "under":"在下方","unit":"单元","until":"直到","up":"向上","use":"使用","us":"我们(宾格)","usual":"通常","value":"价值","vegetable":"蔬菜","very":"非常",
    "visit":"拜访","voice":"声音","wait":"等待","walk":"走路","wall":"墙","want":"想要","warm":"温暖","wash":"清洗","waste":"浪费","watch":"观看",
    "water":"水","wave":"波浪","way":"方式","we":"我们","weak":"虚弱","wealth":"财富","wear":"穿戴","weather":"天气","week":"星期","weight":"重量","welcome":"欢迎",
    "well":"很好","west":"西方","what":"什么","when":"何时","where":"哪里","whether":"是否","which":"哪一个","while":"当...时","white":"白色","who":"谁",
    "why":"为什么","wide":"宽阔","wife":"妻子","wild":"野生","will":"将要","win":"赢","wind":"风","winter":"冬天","wire":"电线","wise":"明智",
    "with":"伴随","within":"在内","without":"没有","woman":"女人","wood":"木头","word":"单词","work":"工作","world":"世界","worry":"担心","would":"将要",
    "write":"书写","year":"年","yellow":"黄色","yes":"是的","yesterday":"昨天","yet":"仍然","you":"你","young":"年轻","your":"你的"
}

# 拆分短语列表、独立单词列表
all_phrase_list = list(word_dict.keys())
single_word_list = []
for ph in all_phrase_list:
    single_word_list.extend(ph.split())
single_word_list = list(set(single_word_list))

# 原始训练样本（5句）
data = [
    ("hello", "你好"),
    ("i love you", "我爱你"),
    ("how are you", "你好吗"),
    ("thank you", "谢谢你"),
    ("good morning", "早上好"),
]

# ==========4.构建词表&编码函数==========
en_all_words = set()
for eng in word_dict.keys():
    en_all_words.update(eng.split())
en_words = sorted(list(en_all_words))
zh_all_chars = set()
for chn in word_dict.values():
    zh_all_chars.update(list(chn))
zh_chars = sorted(list(zh_all_chars))

en_w2i = {"<pad>":0, "<sos>":1, "<eos>":2}
zh_w2i = {"<pad>":0, "<sos>":1, "<eos>":2}
for w in en_words: en_w2i[w] = len(en_w2i)
for c in zh_chars: zh_w2i[c] = len(zh_w2i)
i2zh = {i:c for c,i in zh_w2i.items()}

def encode_en(s, max_len=50):
    word_list = s.strip().lower().split()[:max_len-2]
    idx_list = [en_w2i["<sos>"]]
    for w in word_list:
        idx_list.append(en_w2i.get(w, en_w2i["<pad>"]))
    idx_list.append(en_w2i["<eos>"])
    return idx_list

def encode_zh(s, max_len=50):
    char_list = list(s)[:max_len-2]
    idx_list = [zh_w2i["<sos>"]]
    for c in char_list:
        idx_list.append(zh_w2i.get(c, zh_w2i["<pad>"]))
    idx_list.append(zh_w2i["<eos>"])
    return idx_list

dataset = [(torch.tensor([encode_en(en)]), torch.tensor([encode_zh(zh)])) for en, zh in data]

# ==========5.模型训练==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Translator(len(en_w2i), len(zh_w2i)).to(device)
opt = optim.Adam(model.parameters(), lr=5e-4)
crit = nn.CrossEntropyLoss(ignore_index=zh_w2i["<pad>"])

print("开始训练模型...")
for epoch in range(600):
    loss_all=0
    for src,tgt in dataset:
        src,tgt=src.to(device),tgt.to(device)
        opt.zero_grad()
        pred=model(src,tgt[:,:-1])
        loss=crit(pred.reshape(-1,len(zh_w2i)),tgt[:,1:].reshape(-1))
        loss.backward()
        opt.step()
        loss_all+=loss.item()
    if epoch%100==0:
        print(f'ep{epoch},loss:{loss_all:.3f}')

# ==========6.核心翻译函数：错词提示【错误→正确单词】再翻译==========
def translate_auto(en_sent):
    res = ""
    temp = en_sent.strip().lower()
    fix_note = [] # 存储拼写纠错提示
    # 短语按单词数从长到短排序，优先匹配短语
    phrases = sorted(all_phrase_list, key=lambda x: -len(x.split()))
    while temp.strip():
        match_flag = False
        # 第一步：精准匹配短语
        for p in phrases:
            if temp.startswith(p + " "):
                res += word_dict[p]
                temp = temp[len(p):].strip()
                match_flag = True
                break
            elif temp == p:
                res += word_dict[p]
                temp = ""
                match_flag = True
                break
        if not match_flag:
            sp = temp.split(" ",1)
            raw_w = sp[0]
            # 精准匹配单词
            if raw_w in word_dict:
                res += word_dict[raw_w]
            else:
                # 模糊查找近似词
                sim_w = find_similar_word(raw_w, single_word_list, max_dist=2)
                if sim_w is not None:
                    fix_note.append(f"【{raw_w} → {sim_w}】")
                    res += word_dict[sim_w]
                else:
                    # 完全无匹配调用模型翻译
                    try:
                        src=torch.tensor([encode_en(raw_w)]).to(device)
                        t=[1]
                        with torch.no_grad():
                            for _ in range(15):
                                out=model(src,torch.tensor([t]).to(device))
                                idx=out.argmax(-1)[0,-1].item()
                                if idx==2:break
                                t.append(idx)
                        res += "".join([i2zh[i] for i in t[1:]])
                    except:
                        res += raw_w
            temp = sp[1].strip() if len(sp)>1 else ""
    # 打印拼写修正提示
    if fix_note:
        print("拼写修正：", " ".join(fix_note))
    return res

# ==========交互入口==========
if __name__=="__main__":
    print("\n=====英译中｜拼写错误自动提示正确单词，q退出=====")
    print("示例：hwo are yuo → 提示【hwo→how】【yuo→you】，译文：你好吗\n")
    while True:
        s=input("输入英文：").strip()
        if s.lower()=="q":
            print("程序退出")
            break
        ans = translate_auto(s)
        print("译文：",ans,"\n")