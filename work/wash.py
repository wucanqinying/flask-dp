#"C:\\Users\\27468\\Desktop\\answer.csv"
import csv
import re

patterns = {
    'name': re.compile(r"'name'\s*:\s*'([^']*)'"),
    'time': re.compile(r"'time'\s*:\s*'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'"),
    'question': re.compile(r"'question'\s*:\s*'(.*?)'", re.DOTALL),
    'answer': re.compile(r"'answer'\s*:\s*'(.*?)'", re.DOTALL)
}
#需要的清洗的数据文件的路径和输出的文件名
with open("C:\\Users\\27468\\Desktop\\answer.csv", 'r', encoding='gb18030') as f_in, \
    open('output.csv', 'w', encoding='gb18030', newline='') as f_out:

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
            #print(f"第{row_idx}行数据不完整已过滤")
            continue

        writer.writerow([
            extracted['name'].ljust(10),  # 用户名列也左对齐
            extracted['time'],
            extracted['question'],
            extracted['answer']
        ])