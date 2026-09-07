from pathlib import Path
from tkinter import Tk, filedialog

from pypdf import PdfReader


def pdf_to_txt(pdf_path: str, txt_path: str | None = None):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"找不到文件: {pdf_file}")

    if txt_path is None:
        txt_file = pdf_file.with_suffix(".txt")
    else:
        txt_file = Path(txt_path)

    reader = PdfReader(str(pdf_file))
    pages_text = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages_text.append(f"--- Page {i} ---\n{text}\n")

    txt_file.write_text("\n".join(pages_text), encoding="utf-8")
    print(f"已生成: {txt_file}")
    return str(txt_file)


def select_and_convert_pdfs():
    root = Tk()
    root.withdraw()

    selected_files = filedialog.askopenfilenames(
        title='请选择要转换的 PDF 文件',
        filetypes=[('PDF 文件', '*.pdf'), ('All Files', '*.*')]
    )

    if not selected_files:
        print('未选择任何文件，退出。')
        return []

    converted = []
    for pdf_path in selected_files:
        output = pdf_to_txt(pdf_path)
        converted.append(output)

    print(f'转换完成，共处理 {len(converted)} 个文件。')
    return converted


if __name__ == "__main__":
    select_and_convert_pdfs()