from pathlib import Path
from tkinter import Tk, filedialog
import pandas as pd


def excel_to_csv(excel_path: str, out_dir: str | None = None) -> str:
    """
    将单个Excel文件(xlsx/xls)全部sheet转为csv，多sheet文件名追加sheet名
    :param excel_path: excel文件路径
    :param out_dir: 输出csv目录，None则输出到原excel同目录
    :return: 最后生成的csv文件路径（多sheet返回最后一个）
    """
    excel_file = Path(excel_path)
    if not excel_file.exists():
        raise FileNotFoundError(f"找不到文件: {excel_file}")

    base_stem = excel_file.stem
    if out_dir is None:
        target_dir = excel_file.parent
    else:
        target_dir = Path(out_dir)
        target_dir.mkdir(exist_ok=True)

    xls = pd.ExcelFile(str(excel_file))
    out_file_path = ""
    for sheet in xls.sheet_names:
        df = pd.read_excel(str(excel_file), sheet_name=sheet)
        if len(xls.sheet_names) == 1:
            out_name = f"{base_stem}.csv"
        else:
            out_name = f"{base_stem}_{sheet}.csv"
        csv_file = target_dir / out_name
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"已转换: {csv_file.name}  ({len(df)} 行)")
        out_file_path = str(csv_file)
    return out_file_path


def select_and_convert_excels():
    root = Tk()
    root.withdraw()
    selected_files = filedialog.askopenfilenames(
        title='请选择要转换的 Excel 文件',
        filetypes=[
            ('Excel 文件', '*.xlsx;*.xls'),
            ('All Files', '*.*')
        ]
    )
    if not selected_files:
        print('未选择任何文件，退出。')
        return []

    converted = []
    for excel_path in selected_files:
        output = excel_to_csv(excel_path)
        converted.append(output)

    print(f'转换完成，共处理 {len(converted)} 个Excel文件。')
    return converted


if __name__ == "__main__":
    select_and_convert_excels()
