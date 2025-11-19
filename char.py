import os
import glob
from tqdm import tqdm


def count_unique_characters(data_dir):
    """
    统计指定目录下所有 .txt 文件中的唯一字符集合
    """
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if not txt_files:
        print(f"错误: 在 {data_dir} 下没有找到 .txt 文件。请检查路径或先运行数据准备脚本。")
        return

    print(f"正在扫描 {len(txt_files)} 个标签文件...")

    unique_chars = set()

    for txt_file in tqdm(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # 将字符串中的每个字符加入集合（自动去重）
                # 也可以在这里加过滤逻辑，比如去掉空格: content = content.replace(" ", "")
                unique_chars.update(list(content))
        except Exception as e:
            print(f"读取文件出错: {txt_file}, 错误: {e}")

    # 统计结果
    num_chars = len(unique_chars)
    nb_cls = num_chars + 1  # +1 是为了 CTC Loss 的 Blank Token

    print("-" * 30)
    print(f"统计完成！")
    print(f"唯一字符总数 (Charset Size): {num_chars}")
    print(f"推荐设置 --nb-cls 参数为: {nb_cls}")
    print("-" * 30)

    # (可选) 将字符集保存到文件，方便后续查看或固定字典顺序
    charset_path = os.path.join(os.path.dirname(data_dir), "charset.txt")
    with open(charset_path, 'w', encoding='utf-8') as f:
        # 排序后写入，保证顺序一致
        sorted_chars = sorted(list(unique_chars))
        for char in sorted_chars:
            f.write(char + "\n")
    print(f"字符表已保存至: {charset_path}")


if __name__ == "__main__":
    # 这里指向你之前脚本生成的 lines 文件夹路径
    lines_dir = "./data/icdar2013/images"

    count_unique_characters(lines_dir)