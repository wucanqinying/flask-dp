import pandas as pd
from collections import Counter
import jieba
import re

# 定义二维性格变量
personality_dimensions = {
    '理性-感性': {
        '理性': ['逻辑', '清晰', '事实', '数据', '分析', '论证', '推理', '客观', '实证', '科学', '理论', '研究', '证据', '严谨', '辩证'],
        '感性': ['情感', '感受', '直觉', '情绪', '体验', '感动', '共鸣', '温暖', '心灵', '柔软', '感性', '情怀', '共情', '触动']
    },
    '幽默-严肃': {
        '幽默': ['调侃', '有趣', '好笑', '夸张', '玩梗', '段子', '搞笑', '幽默', '滑稽', '讽刺', '吐槽', '哈哈哈', '笑哭', '狗头', '捂脸'],
        '严肃': ['认真', '正式', '庄重', '严谨', '严肃', '重要', '深刻', '沉重', '正式', '正经', '认真', '责任', '思考', '讨论']
    },
    '务实-理想': {
        '务实': ['实用', '性价比', '实际', '简单', '可行', '现实', '落地', '操作', '执行', '效率', '成本', '收益', '务实', '实践'],
        '理想': ['梦想', '理想', '愿景', '未来', '幻想', '乌托邦', '美好', '追求', '憧憬', '远方', '希望', '信念', '理想主义']
    },
    '耐心-急躁': {
        '耐心': ['详细', '解释', '逐步', '拆解', '耐心', '细致', '慢慢', '循序渐进', '讲解', '教导', '引导', '宽容', '理解'],
        '急躁': ['快速', '着急', '匆忙', '不耐烦', '急躁', '焦虑', '紧迫', '急于', '赶时间', '急迫', '心急', '匆忙']
    },
    '直率-委婉': {
        '直率': ['直接', '简洁', '干脆', '直率', '坦率', '直言', '直白', '开门见山', '不拐弯抹角', '直截了当', '爽快'],
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
df = pd.read_excel("C:/Users/DELL/Desktop/大数据竞赛/output_处理后.xlsx")

# 加载停用词
stopwords = load_stopwords("C:/Users/DELL/Desktop/大数据竞赛/baidu_stopwords.txt")

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
output_df.to_excel('user_personalities_2d.xlsx', index=False)

# 读取Excel文件
file_path = 'user_personalities_2d.xlsx'
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