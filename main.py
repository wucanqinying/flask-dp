import flask
from flask import *

app = Flask(__name__,static_url_path='/static',static_folder='static',template_folder='templates')

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/install',methods=['GET','POST'])
def install():

    import subprocess
    import sys
    import pkg_resources

    # 定义需要安装的库及其版本（可根据需求修改）
    REQUIRED_LIBRARIES = {
        'pandas': '2.2.1',
        'jieba': '0.42.1',
        'Drissionpage': '4.1.0.15',
        'BeautifulSoup4': '4.12.3',
        'flask': '2.3.3',
        'click': '8.1.7',
        'pymysql':  '1.1.1',
        'matplotlib': '3.8.3',
        'numpy': '1.26.4'
    }

    def install_library(library, version=None):
        """安装指定的 Python 库"""
        try:
            # 检查库是否已安装
            pkg_resources.get_distribution(library)
            print(f"{library} 已安装")
        except pkg_resources.DistributionNotFound:
            # 未安装，执行安装
            print(f"正在安装 {library}...")
            package = f"{library}=={version}" if version else library
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{library} 安装完成")

    def main():
        print("开始检查和安装 Python 库...")
        for lib, version in REQUIRED_LIBRARIES.items():
            install_library(lib, version)
        print("所有库检查和安装完成！")

    if __name__ == "__main__":
        main()

    return render_template('cs.html')

@app.route('/paqu',methods=['GET','POST'])
def paqu():
    return render_template('paqu.html')

@app.route('/caiji',methods=['GET','POST'])
def caiji():
    from datetime import timedelta
    import pymysql
    from DrissionPage import ChromiumPage
    from DrissionPage import ChromiumOptions
    import time
    import csv
    from random import random
    import re
    from bs4 import BeautifulSoup
    import pandas as pd
    from openpyxl.styles import Alignment


    url = request.form.get('url')
    question = request.form.get('question')
    location = request.form.get('location')



    if request.form.get('times'):
        rolltime = int(request.form.get('times'))
    else:
        rolltime = 10

    path = r"Application\chrome.exe"

    co = ChromiumOptions()
    co.set_browser_path(path)


    if request.form.get('flag1'):
        flag1 = int(request.form.get('flag1'))
        if flag1 == 1:
            co.headless()
    else:
        co.headless(False)


    dp = ChromiumPage(co)


    try:

        uri = []
        pattern = r"(?<=/people/)[a-z0-9-]+"

        match = re.search(pattern, url)
        if match:
            uri.append(match.group())

        for name in uri:

            listen_url = "https://www.zhihu.com/api/v4/members/%s/answers?" % (name)

            dp.listen.start(listen_url)

            dp.get("https://www.zhihu.com/people/%s/answers" % (name))

            list = []
            time.sleep(2)

            for i in range(rolltime):

                try:
                    resp = dp.listen.wait()

                    if resp.response.body:

                        json_data = resp.response.body

                        links = json_data['data']

                        for index in links:
                            text = index['content']

                            soup = BeautifulSoup(text, 'html.parser')

                            clean_text = soup.get_text(separator=' ')

                            dit = {
                                'name': index['author']['name'],

                                'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(index['created_time']))),

                                'question': index['question']['title'],

                                'answer': clean_text,

                            }
                            print(dit)
                            list.append(dit)

                        current_height = dp.run_js('return document.body.scrollHeight;')

                        dp.run_js("window.scrollTo(0, document.body.scrollHeight);")

                        #time.sleep(random() * 6)

                        dp.wait.load_start(timeout=2)

                        new_height = dp.run_js('return document.body.scrollHeight;')

                        if new_height == current_height:
                            print("页面高度不再变化，已到达底部或加载完成！")
                            break

                        last_height = new_height

                    else:
                        break

                except:
                    print('出错了')
                    break
                    pass



            output = r'work\%s.csv' % (name)

            with open(output, mode='w', encoding='gb18030', newline='', errors='replaces') as f:
                for i in list:
                    i = str(i)
                    f.write(i)
                    f.write('\n')

            patterns = {
                'name': re.compile(r"'name'\s*:\s*'([^']*)'"),
                'time': re.compile(r"'time'\s*:\s*'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'"),
                'question': re.compile(r"'question'\s*:\s*'(.*?)'", re.DOTALL),
                'answer': re.compile(r"'answer'\s*:\s*'(.*?)'", re.DOTALL)
            }

            # 创建数据容器
            cleaned_data = []
            # 需要清洗的文件路径
            with open(output, 'r', encoding='gb18030') as f_in:

                reader = csv.reader(f_in)

                for row_idx, row in enumerate(reader, 1):
                    if not row:
                        continue

                    extracted = {key: '' for key in patterns.keys()}
                    valid = True

                    try:
                        for col_idx, key in enumerate(['name', 'time', 'question', 'answer']):
                            if len(row) > col_idx and row[col_idx].strip():
                                match = patterns[key].search(row[col_idx])
                                if match:
                                    extracted[key] = match.group(1).strip().replace('\n', ' ')
                                else:
                                    if key in ['question', 'answer']:
                                        valid = False
                            else:
                                if key in ['question', 'answer']:
                                    valid = False

                    except Exception as e:
                        print(f"第{row_idx}行解析失败：{str(e)}")
                        continue

                    if not valid or not extracted['question'] or not extracted['answer']:
                        continue

                    # 收集有效数据
                    cleaned_data.append([
                        extracted['name'],
                        extracted['time'],
                        extracted['question'],
                        extracted['answer']
                    ])

            # 转换为DataFrame并导出Excel
            df = pd.DataFrame(cleaned_data, columns=['用户名', '时间', '问题内容', '答案内容'])

            # 使用ExcelWriter设置格式
            with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
                df.to_excel(writer, index=False)

                # 获取工作表对象
                worksheet = writer.sheets['Sheet1']

                # 设置列宽和对齐方式
                column_widths = [15, 20, 50, 70]  # 根据需求调整
                for col_idx, width in enumerate(column_widths, 1):
                    worksheet.column_dimensions[chr(64 + col_idx)].width = width

                # 设置左对齐
                for row in worksheet.iter_rows(min_row=2):
                    for cell in row[:2]:

                        cell.alignment = Alignment(horizontal='left')

        dp.quit()


    except:
        dp.quit()

    return render_template('chuli.html')

@app.route('/miaohui',methods=['GET','POST'])
def miaohui():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import jieba
    import re
    from pathlib import Path
    from collections import defaultdict

    # ===== 1. 定义15个跨领域兴趣维度 =====
    interest_dimensions = {
        # 科学技术类
        "自然科学": ["物理", "化学", "生物学", "天文", "地质", "量子", "相对论", "宇宙", "DNA", "进化论", "实验室"],
        "工程技术": ["机械", "电子", "自动化", "机器人", "材料", "能源", "建筑", "土木", "航空航天", "3D打印"],

        # 人文社科类
        "文学创作": ["小说", "诗歌", "散文", "作家", "诺贝尔文学奖", "写作", "出版社", "经典", "名著", "科幻"],
        "历史考古": ["朝代", "文明", "古埃及", "罗马", "考古", "文物", "博物馆", "二战", "中世纪", "工业革命"],
        "哲学心理": ["哲学", "尼采", "康德", "心理学", "弗洛伊德", "认知", "意识", "存在主义", "伦理学", "形而上学"],

        # 商业经济类
        "商业管理": ["创业", "CEO", "战略", "领导力", "MBA", "商业模式", "市场营销", "品牌", "供应链", "组织行为"],
        "金融投资": ["股票", "基金", "比特币", "区块链", "期货", "外汇", "财报", "估值", "巴菲特", "华尔街"],
        "宏观经济": ["GDP", "通货膨胀", "货币政策", "美联储", "经济周期", "贸易战", "全球化", "人口红利", "供给侧"],

        # 艺术娱乐类
        "影视艺术": ["电影", "导演", "奥斯卡", "Netflix", "剧本", "表演", "纪录片", "电影节", "影评", "好莱坞"],
        "音乐舞蹈": ["古典乐", "钢琴", "交响乐", "摇滚", "爵士", "演唱会", "编曲", "声乐", "芭蕾", "街舞"],
        "游戏动漫": ["电竞", "Steam", "任天堂", "原神", "二次元", "漫威", "DC", "Cosplay", "剧情", "角色设计"],

        # 生活健康类
        "健康养生": ["健身", "瑜伽", "冥想", "营养", "维生素", "睡眠", "抗衰老", "保健品", "中医", "针灸"],
        "旅行户外": ["自驾游", "背包客", "登山", "潜水", "露营", "国家公园", "民宿", "环球旅行", "摄影", "极光"],
        "美食烹饪": ["食谱", "米其林", "烘焙", "咖啡", "茶道", "火锅", "寿司", "葡萄酒", "食材", "分子料理"],

        # 社会与教育
        "教育学习": ["大学", "在线课程", "MOOC", "终身学习", "记忆力", "学习方法", "高考", "留学", "雅思", "教科书"],
        "社会热点": ["环保", "碳中和", "元宇宙", "Web3", "AI伦理", "性别平等", "老龄化", "乡村振兴", "公共政策"],
    }

    # ===== 2. 辅助函数 =====
    def load_stopwords(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)

    def filter_stopwords(text, stopwords):
        words = jieba.lcut(text)
        return ' '.join(word for word in words if word not in stopwords)

    def contains_keyword(text, keywords):
        return any(re.search(rf'\b{re.escape(keyword)}\b', text) for keyword in keywords)

    # ===== 3. 分析用户兴趣 =====
    def analyze_interests(input_excel, stopwords_file, output_excel):
        df = pd.read_excel(input_excel)
        stopwords = load_stopwords(stopwords_file)
        user_interests = defaultdict(lambda: defaultdict(int))

        for _, row in df.iterrows():
            user_id = row['用户ID']
            comment = str(row['评论']) if pd.notna(row['评论']) else ''
            filtered_comment = filter_stopwords(comment, stopwords)

            for interest, keywords in interest_dimensions.items():
                if contains_keyword(filtered_comment, keywords):
                    user_interests[user_id][interest] += 1

        # 保存结果
        output_df = pd.DataFrame([
            {"用户ID": user_id, **interests}
            for user_id, interests in user_interests.items()
        ])
        output_df.to_excel(output_excel, index=False)
        return user_interests

    # ===== 4. 绘制雷达图（优化版） =====
    def draw_radar_chart(user_id, scores, output_dir):
        categories = list(interest_dimensions.keys())
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, scores, 'o-', linewidth=2, color='#1E90FF')
        ax.fill(angles, scores, alpha=0.25, color='#1E90FF')

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
        ax.set_rlabel_position(30)
        plt.title(f'用户【{user_id}】的跨领域兴趣分析', size=16, pad=20)

        output_path = Path(output_dir) / f"{user_id}.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # ===== 5. 主程序 =====
    if __name__ == "__main__":
        input_excel = r"./work/output.xlsx"
        stopwords_file = r"work\baidu_stopwords.txt"  # 停用词表路径
        output_excel = r"workout\user_interests_15d.xlsx"
        output_dir = "workout"

        # 分析兴趣
        user_interests = analyze_interests(input_excel, stopwords_file, output_excel)

        # 绘制雷达图
        for user_id, interests in user_interests.items():
            scores = [interests.get(interest, 0) for interest in interest_dimensions.keys()]
            draw_radar_chart(user_id, scores, output_dir)

        print(f"✅ 分析完成！\n- Excel结果: {output_excel}\n- 雷达图目录: {output_dir}")

    return render_template("xingge.html")

@app.route("/xingge",methods=["GET", "POST"])
def xingge():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import jieba
    import re
    from pathlib import Path

    # 定义二维性格变量
    personality_dimensions = {
        '理性-感性': {
            '理性': ['逻辑', '清晰', '事实', '数据', '分析', '论证', '推理', '客观', '实证', '科学', '理论', '研究',
                     '证据', '严谨', '辩证'],
            '感性': ['情感', '感受', '直觉', '情绪', '体验', '感动', '共鸣', '温暖', '心灵', '柔软', '感性', '情怀',
                     '共情', '触动']
        },
        '幽默-严肃': {
            '幽默': ['调侃', '有趣', '好笑', '夸张', '玩梗', '段子', '搞笑', '幽默', '滑稽', '讽刺', '吐槽', '哈哈哈',
                     '笑哭', '狗头', '捂脸'],
            '严肃': ['认真', '正式', '庄重', '严谨', '严肃', '重要', '深刻', '沉重', '正式', '正经', '认真', '责任',
                     '思考', '讨论']
        },
        '务实-理想': {
            '务实': ['实用', '性价比', '实际', '简单', '可行', '现实', '落地', '操作', '执行', '效率', '成本', '收益',
                     '务实', '实践'],
            '理想': ['梦想', '理想', '愿景', '未来', '幻想', '乌托邦', '美好', '追求', '憧憬', '远方', '希望', '信念',
                     '理想主义']
        },
        '耐心-急躁': {
            '耐心': ['详细', '解释', '逐步', '拆解', '耐心', '细致', '慢慢', '循序渐进', '讲解', '教导', '引导', '宽容',
                     '理解'],
            '急躁': ['快速', '着急', '匆忙', '不耐烦', '急躁', '焦虑', '紧迫', '急于', '赶时间', '急迫', '心急', '匆忙']
        },
        '直率-委婉': {
            '直率': ['直接', '简洁', '干脆', '直率', '坦率', '直言', '直白', '开门见山', '不拐弯抹角', '直截了当',
                     '爽快'],
            '委婉': ['含蓄', '委婉', '间接', '暗示', '婉转', '拐弯抹角', '含蓄', '隐晦', '暗示', '绕弯子', '含蓄']
        },
        '温和-强硬': {
            '温和': ['平和', '友好', '亲和', '温和', '温柔', '和蔼', '友善', '宽容', '理解', '体贴', '耐心', '柔软'],
            '强硬': ['强硬', '激烈', '冲突', '对抗', '坚决', '坚定', '强势', '激烈', '不容置疑', '不容反驳', '强硬']
        }
    }

    # 读取停用词文件
    def load_stopwords(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            stopwords = set(line.strip() for line in file)
        return stopwords

    # 过滤停用词
    def filter_stopwords(text, stopwords):
        words = jieba.lcut(text)  # 使用 jieba 进行中文分词
        return ' '.join(word for word in words if word not in stopwords)

    # 关键词匹配
    def contains_keyword(text, keywords):
        return any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords)

    # 读取Excel文件
    df = pd.read_excel("./work/output.xlsx")

    # 加载停用词
    stopwords = load_stopwords("./work/baidu_stopwords.txt")

    # 初始化用户性格字典
    user_personalities = {}

    # 分析每个用户的评论
    for index, row in df.iterrows():
        user_id = row['用户ID']
        comment = str(row['评论']) if pd.notna(row['评论']) else ''  # 处理空值

        # 过滤停用词
        filtered_comment = filter_stopwords(comment, stopwords)

        if user_id not in user_personalities:
            user_personalities[user_id] = {dim: {'倾向': None, '得分': 0} for dim in personality_dimensions.keys()}

        # 计算每个二维性格变量的倾向
        for dimension, traits in personality_dimensions.items():
            for trait, keywords in traits.items():
                if contains_keyword(filtered_comment, keywords):
                    # 根据关键词匹配结果更新得分
                    if trait == dimension.split('-')[0]:  # 第一个维度
                        user_personalities[user_id][dimension]['得分'] += 1
                    else:  # 第二个维度
                        user_personalities[user_id][dimension]['得分'] -= 1

        # 确定每个二维变量的倾向
        for dimension in personality_dimensions.keys():
            if user_personalities[user_id][dimension]['得分'] > 0:
                user_personalities[user_id][dimension]['倾向'] = dimension.split('-')[0]  # 第一个维度
            elif user_personalities[user_id][dimension]['得分'] < 0:
                user_personalities[user_id][dimension]['倾向'] = dimension.split('-')[1]  # 第二个维度
            else:
                user_personalities[user_id][dimension]['倾向'] = '中性'  # 得分相等时为中性

    # 输出用户性格倾向
    for user_id, dimensions in user_personalities.items():
        print(f"用户ID: {user_id}")
        for dimension, result in dimensions.items():
            print(f"{dimension}: {result['倾向']} (得分: {result['得分']})")
        print()

    # 将结果保存到新的Excel文件
    output_data = []
    for user_id, dimensions in user_personalities.items():
        row = {'用户ID': user_id}
        for dimension, result in dimensions.items():
            row[dimension] = result['倾向']
            row[f"{dimension}得分"] = result['得分']
        output_data.append(row)

    output_df = pd.DataFrame(output_data)
    output_df.to_excel('./workout/user_personalities_2d.xlsx', index=False)

    # 读取Excel文件
    file_path = './workout/user_personalities_2d.xlsx'
    df = pd.read_excel(file_path)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 定义雷达图的维度（6个维度）
    categories = ['理性-感性', '幽默-严肃', '务实-理想', '耐心-急躁', '直率-委婉', '温和-强硬']
    N = len(categories)

    # 为每个用户创建雷达图
    for index, row in df.iterrows():
        user_id = row['用户ID']

        # 获取6个维度的得分
        values = [
            row['理性-感性得分'],
            row['幽默-严肃得分'],
            row['务实-理想得分'],
            row['耐心-急躁得分'],
            row['直率-委婉得分'],
            row['温和-强硬得分']
        ]

        # 由于雷达图需要闭合，所以将第一个值重复在末尾
        values += values[:1]

        # 计算每个角度的角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # 初始化雷达图
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        # 绘制数据
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        # 设置标签
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)

        # 设置标题
        plt.title(f'用户 {user_id} 的性格特质雷达图', size=16, y=1.1)

        # 保存图像
        plt.tight_layout()
        output_path = Path(f'radar_charts/{user_id}_radar.png')
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        print(f'已为用户 {user_id} 生成雷达图并保存到 {output_path}')

    print("所有用户的雷达图已生成完毕！")
@app.route("/answer",methods=["POST"])
def answer():
    import pandas as pd
    import numpy as np
    import jieba
    from sklearn.preprocessing import MinMaxScaler

    # 加载数据
    df = pd.read_excel(r'./workout/user_personalities_2d.xlsx')
    score_columns = ['理性-感性得分', '幽默-严肃得分', '务实-理想得分',
                     '耐心-急躁得分', '直率-委婉得分', '温和-强硬得分']

    # Min-Max标准化用户数据
    scaler = MinMaxScaler()
    user_scores_normalized = scaler.fit_transform(df[score_columns])

    # 扩展后的关键词库（示例部分）
    dimension_keywords = {
        '理性-感性': {
            'positive': [
                # 学术研究类
                '数据', '分析', '统计', '逻辑', '实证', '模型', '算法', '验证', '量化', '公式',
                '定理', '推导', '标准差', '回归分析', '假设检验', '因果', '论证', '框架', '方法论',
                '指标体系', '信效度', '抽样', '变量', '对照组', '双盲实验', '显著性', '协方差',
                '方差分析', '贝叶斯', '神经网络', '架构', '流程图', '接口', '模块化', '调试', '优化',
                '迭代', '封装', '自动化', '鲁棒性', '冗余', '容错率', '基准测试', '压力测试', '正交实验',
                '收敛性', '蒙特卡洛', '启发式', '时间复杂度', '空间复杂度', '可证伪性'
            ],
            'negative': [
                # 情感与主观体验
                '感受', '直觉', '共情', '感动', '流泪', '心痛', '温暖', '治愈', '情怀', '意境',
                '诗意', '浪漫', '情调', '触景生情', '睹物思人', '多愁善感', '美感', '韵律', '旋律',
                '构图', '写意', '蒙太奇', '意识流', '即兴', '灵感', '创作', '审美', '抽象派', '印象派',
                '超现实主义', '我觉得', '我相信', '我感受到', '个人认为', '私以为', '第六感', '心灵感应',
                '缘分', '命运', '灵魂', '禅意', '宿命', '直觉判断', '情感共鸣', '心灵震撼', '感同身受',
                '怦然心动', '潸然泪下', '怅然若失', '五味杂陈'
            ]
        },
        '幽默-严肃': {
            'positive': [
                # 幽默表达类
                '搞笑', '段子', '哈哈哈', '幽默', '趣闻', '笑话', '滑稽', '吐槽', '玩梗', '神回复',
                '谐音梗', '冷幽默', '反讽', '自嘲', '表情包', '恶搞', '脑洞', '无厘头', '反转', '调侃',
                '捧哏', '逗比', '笑喷', '笑死', '笑场', '爆笑', '妙啊', '沙雕', '鬼畜', '名场面',
                '谐星', '欢乐', '轻松', '趣味', '诙谐', '滑稽戏', '脱口秀', '相声', '小品', '模仿秀',
                '打油诗', '网络流行语', '颜文字', '熊猫头', '金句', '梗图', '玩坏', '灵魂拷问', '人间真实'
            ],
            'negative': [
                # 严肃讨论类
                '严肃', '认真', '探讨', '研究', '分析', '讨论', '论证', '学术', '严谨', '考据',
                '文献', '引证', '理论', '原理', '方法论', '逻辑链', '数据支持', '实验设计', '辩证', '批判',
                '反思', '深度', '本质', '根源', '系统性', '复杂性', '哲学', '思辨', '综述', '命题',
                '假说', '验证', '推导', '定理', '定律', '公式', '模型', '框架', '指标体系', '方法论',
                '学术规范', '参考文献', '同行评议', '量化分析', '实证研究', '双盲测试', '可重复性', '学术伦理'
            ]
        },
        '务实-理想': {
            'positive': [
                # 务实操作类
                '实用', '应用', '方法', '步骤', '操作', '经验', '案例', '建议', '教程', '手册',
                '指南', '攻略', '技巧', '窍门', '工具', '资源', '模板', '清单', '流程图', '时间表',
                '预算', '成本', '效率', '产出', '落地', '执行', '实施', '反馈', '改进', '迭代',
                '测试', '调试', '优化', '维护', '故障排除', '文档', '用户手册', 'SOP', 'KPI', 'ROI',
                '性价比', '快速上手', '即插即用', '开箱即用', '最佳实践', '避坑指南', '经验分享', '避雷', '实测'
            ],
            'negative': [
                # 理想愿景类
                '理想', '愿景', '梦想', '未来', '理念', '乌托邦', '蓝图', '设想', '可能性', '变革',
                '颠覆', '创新', '突破', '范式转移', '终极', '完美', '乌托邦', '理想国', '大同', '终极关怀',
                '哲学思考', '人类命运', '星辰大海', '探索未知', '无限可能', '永续发展', '终极答案', '形而上学',
                '本体论', '认识论', '价值判断', '存在主义', '自由意志', '道德律令', '普世价值', '终极目标',
                '理想主义', '乌托邦主义', '完美主义', '终极解决方案', '革命性', '范式革新', '重新定义', '颠覆性创新',
                '从0到1', '第二曲线', '升维思考', '元问题', '第一性原理'
            ]
        },
        '耐心-急躁': {
            'positive': [
                # 耐心培养类
                '耐心', '详细', '步骤', '指导', '解释', '帮助', '支持', '循序渐进', '分阶段', '分步骤',
                '手把手', '零基础', '小白友好', '入门指南', '常见问题', '注意事项', '避坑提示', '长期主义',
                '持续改进', '迭代优化', '反复练习', '熟能生巧', '温故知新', '阶段性总结', '定期反馈', '检查点',
                '里程碑', '时间管理', '番茄工作法', 'GTD', '任务分解', '优先级', '抗压能力', '延迟满足',
                '心流状态', '正念练习', '情绪管理', '压力缓解', '深呼吸', '冥想', '瑜伽', '渐进式', '分层次',
                '模块化学习', '知识体系', '系统化', '结构化', '思维导图', '复盘', 'PDCA循环'
            ],
            'negative': [
                # 急躁表现类
                '尽快', '急', '马上', '快', '速度', '赶紧', '紧急', '立刻', '立即', '火烧眉毛',
                '迫在眉睫', '刻不容缓', '十万火急', '赶时间', '没耐心', '跳步', '走捷径', '偷工减料',
                '敷衍了事', '三分钟热度', '浅尝辄止', '虎头蛇尾', '半途而废', '急躁', '浮躁', '焦虑',
                '压力山大', '崩溃', '抓狂', '原地爆炸', '心态炸裂', '没时间', '来不及', '截止日期',
                '倒计时', 'DDL', '通宵', '熬夜', '加班', '连轴转', '多任务', '分身乏术', '应接不暇',
                '手忙脚乱', '顾此失彼', '时间不够', '效率低下', '拖延症', '最后一刻'
            ]
        },
        '直率-委婉': {
            'positive': [
                # 直率表达类
                '必须', '显然', '确实', '当然', '无疑', '肯定', '绝对', '明确', '直接', '坦率',
                '直言不讳', '开门见山', '一针见血', '毫不掩饰', '直截了当', '单刀直入', '正面回应', '明确表态',
                '斩钉截铁', '毋庸置疑', '无可争辩', '板上钉钉', '铁证如山', '实锤', '实打实', '赤裸裸',
                '血淋淋', '尖锐', '犀利', '戳破', '揭穿', '打脸', '啪啪响', '不留情面', '撕破脸',
                '捅破窗户纸', '说破无毒', '直球', '硬刚', '正面硬怼', '不绕弯子', '有一说一', '明人不说暗话',
                '打开天窗说亮话', '话糙理不糙', '直给', '不藏着掖着'
            ],
            'negative': [
                # 委婉表达类
                '或许', '可能', '也许', '大概', '建议', '考虑', '不妨', '或者', '说不定', '说不定',
                '某种程度上', '某种意义上', '个人拙见', '仅供参考', '可能欠妥', '恐有不周', '有待商榷', '尚存疑问',
                '可能存在', '不排除', '或许可以', '从长计议', '从轻发落', '留有余地', '委婉', '含蓄', '暗示',
                '旁敲侧击', '弦外之音', '话里有话', '点到为止', '心照不宣', '看破不说破', '顾全大局', '留面子',
                '给台阶', '拐弯抹角', '绕圈子', '打太极', '和稀泥', '模棱两可', '含糊其辞', '避重就轻', '转移话题',
                '春秋笔法', '外交辞令', '场面话', '客套话', '官样文章', '虚与委蛇'
            ]
        },
        '温和-强硬': {
            'positive': [
                # 温和沟通类
                '建议', '考虑', '可以', '或许', '可能', '灵活', '协商', '讨论', '探讨', '商量',
                '请教', '请教', '劳驾', '麻烦', '辛苦', '感谢', '感恩', '包容', '理解', '体谅',
                '换位思考', '将心比心', '求同存异', '和而不同', '各退一步', '折中方案', '双赢', '共赢',
                '友好协商', '和平共处', '互相尊重', '平等对话', '理性讨论', '心平气和', '温文尔雅', '彬彬有礼',
                '以理服人', '循循善诱', '春风化雨', '润物无声', '点到为止', '留有余地', '适可而止', '见好就收',
                '得饶人处且饶人', '退一步海阔天空', '和气生财', '以柔克刚', '刚柔并济', '柔中带刚'
            ],
            'negative': [
                # 强硬态度类
                '必须', '绝对', '一定', '肯定', '严厉', '严格', '坚决', '绝不', '毫无', '完全',
                '彻底', '百分之百', '毋庸置疑', '没商量', '死命令', '硬性规定', '铁律', '红线', '底线',
                '零容忍', '一票否决', '强制', '强迫', '逼迫', '威胁', '恐吓', '最后通牒', '没得谈',
                '不容置疑', '说一不二', '独断专行', '刚愎自用', '固执己见', '一意孤行', '强词夺理', '咄咄逼人',
                '针锋相对', '寸步不让', '以势压人', '仗势欺人', '得理不饶人', '赶尽杀绝', '不留余地', '撕破脸',
                '鱼死网破', '玉石俱焚', '你死我活', '不共戴天', '势不两立', '血战到底'
            ]
        }
    }

    def analyze_question(text):
        words = list(jieba.cut(text))
        feature_vector = []
        for dim in score_columns:
            dim_name = dim.replace('得分', '')
            pos = sum(1 for w in words if w in dimension_keywords[dim_name]['positive'])
            neg = sum(1 for w in words if w in dimension_keywords[dim_name]['negative'])
            feature_vector.append(pos - neg)
        return np.array(feature_vector)

    def recommend_users(question_vector):
        # 归一化向量
        q_norm = question_vector / np.linalg.norm(question_vector) if np.linalg.norm(
            question_vector) != 0 else question_vector
        similarities = []
        for user in user_scores_normalized:
            u_norm = user / np.linalg.norm(user)
            similarities.append(np.dot(q_norm, u_norm))
        df['相似度'] = similarities
        return df.sort_values('相似度', ascending=False).head(5)[['用户ID', '相似度'] + score_columns]

    if __name__ == "__main__":
        qa = request.form.get('question')
        questions = qa
        vec = analyze_question(questions)
        print(f"\n问题：{questions}")
        a = recommend_users(vec)[['用户ID', '相似度']].to_string(index=False)
        print(type(a))
        print(a)

        return render_template("answer.html",question=questions,answer=a)

@app.errorhandler(404)
def page_not_found(e):
    return "404!!!你出错了",404
    pass


if __name__ == '__main__':
    app.run(host="127.0.0.1",port=8888,debug=True)