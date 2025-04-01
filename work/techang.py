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
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
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
    input_excel = "C:/Users/DELL/Desktop/大数据竞赛/output_处理后.xlsx"  # 替换为你的输入文件
    stopwords_file = "C:/Users/DELL/Desktop/大数据竞赛/baidu_stopwords.txt"  # 停用词表路径
    output_excel = "user_interests_15d.xlsx"
    output_dir = "interest_radar_charts"
    
    # 分析兴趣
    user_interests = analyze_interests(input_excel, stopwords_file, output_excel)
    
    # 绘制雷达图
    for user_id, interests in user_interests.items():
        scores = [interests.get(interest, 0) for interest in interest_dimensions.keys()]
        draw_radar_chart(user_id, scores, output_dir)
    
    print(f"✅ 分析完成！\n- Excel结果: {output_excel}\n- 雷达图目录: {output_dir}")