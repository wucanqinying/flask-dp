import flask
#from docutils.nodes import option
from flask import *
from datetime import timedelta

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
        'jieba': '0.42.1',
        'Drissionpage': '4.1.0.15',
        'BeautifulSoup4': '4.12.3',
        'flask': '2.3.3',
        'click': '8.1.7',
        'pymysql':  '1.1.1',
        'pandas': '2.2.1',  
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
    #import pymysql
    from DrissionPage import ChromiumPage
    from DrissionPage import ChromiumOptions
    import time
    import csv
    from random import random
    import re
    from bs4 import BeautifulSoup
    
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

            with open(output, 'r', encoding='gb18030') as f_in, \
                open(r'work\output.csv', 'a', encoding='gb18030', newline='') as f_out:

                reader = csv.reader(f_in)
                writer = csv.writer(f_out)

                # 写入带格式表头
                writer.writerow(['用户名'.ljust(10), '时间'.ljust(20), '问题内容', '答案内容'])

                for row_idx, row in enumerate(reader, 1):
                    if not row:
                        continue

                    extracted = {key: '' for key in patterns.keys()}
                    valid = True  # 数据有效性标记

                    try:
                        for col_idx, key in enumerate(['name', 'time', 'question', 'answer']):
                            if len(row) > col_idx and row[col_idx].strip():
                                match = patterns[key].search(row[col_idx])
                                if match:
                                    # 特殊处理时间列左对齐
                                    if key == 'time':
                                        extracted[key] = match.group(1).ljust(20)  # 固定20字符宽度左对齐
                                    else:
                                        extracted[key] = match.group(1).strip().replace('\n', ' ')
                                else:
                                    if key in ['question', 'answer']:  # 关键列缺失匹配
                                        valid = False
                            else:
                                if key in ['question', 'answer']:  # 关键列不存在
                                    valid = False
                    except Exception as e:
                        print(f"第{row_idx}行解析失败：{str(e)}")
                        continue

                    # 空白内容过滤逻辑
                    if not valid or not extracted['question'] or not extracted['answer']:
                        # print(f"第{row_idx}行数据不完整已过滤")
                        continue

                    writer.writerow([
                        extracted['name'].ljust(10),  # 用户名列也左对齐
                        extracted['time'],
                        extracted['question'],
                        extracted['answer']
                    ])
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
        input_excel = r"work\output.xlsx"  # 替换为你的输入文件
        stopwords_file = r"work\baidu_stopwords.txt"  # 停用词表路径
        output_excel = r"workout\user_interests_15d.xlsx"
        output_dir = "interest_radar_charts"

        # 分析兴趣
        user_interests = analyze_interests(input_excel, stopwords_file, output_excel)

        # 绘制雷达图
        for user_id, interests in user_interests.items():
            scores = [interests.get(interest, 0) for interest in interest_dimensions.keys()]
            draw_radar_chart(user_id, scores, output_dir)

        print(f"✅ 分析完成！\n- Excel结果: {output_excel}\n- 雷达图目录: {output_dir}")



@app.errorhandler(404)
def page_not_found(e):
    return "404!!!你出错了",404
    pass


if __name__ == '__main__':
    app.run(host="127.0.0.1",port=8888,debug=True)
