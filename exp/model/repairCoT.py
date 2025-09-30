
import json
import re

filepath = "/fs-computility/niuyazhe/wangjieyi/code/r1-aqa-main-v1/exp/model/test_400/res_mmau_mini_CoT.json"
out_file_path = "/fs-computility/niuyazhe/wangjieyi/code/r1-aqa-main-v1/exp/model/test_400/res_mmau_mini_CoT2.json"

with open(filepath, 'r', encoding='utf-8') as f, open(out_file_path, "w",encoding="utf-8") as writer:
    data = json.load(f)
    new_data = []
    for line in data:
        text = line["model_prediction"]
        # 使用正则表达式提取<answer>标签中的内容
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            ans = match.group(1)  # 输出: Man
        else:
            if "<answer>" in text:
                print("No match found")
                dic = text.split("<answer>")
                dic2 = dic[1].split("</answer>")
                ans = dic2[0]
            else:
                ans = ""
        ans = ans.strip()
        line["model_prediction"] = ans
        new_data.append(line)
    json.dump(new_data, writer, indent=4, ensure_ascii=False)