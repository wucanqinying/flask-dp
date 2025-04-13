# 新增依赖库
import pandas as pd
from openpyxl.styles import Alignment

# "C:\\Users\\27468\\Desktop\\answer.csv"
import csv
import re

patterns = {
    'name': re.compile(r"'name'\s*:\s*'([^']*)'"),
    'time': re.compile(r"'time'\s*:\s*'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'"),
    'question': re.compile(r"'question'\s*:\s*'(.*?)'", re.DOTALL),
    'answer': re.compile(r"'answer'\s*:\s*'(.*?)'", re.DOTALL)
}

# 创建数据容器
cleaned_data = []
#需要清洗的文件路径
with open("C:\\Users\\27468\\Desktop\\answer.csv", 'r', encoding='gb18030') as f_in:
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
        for cell in row[:2]:  # 前两列左对齐
            cell.alignment = Alignment(horizontal='left')