from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import TableItem, PictureItem


pdf_options = PdfPipelineOptions()
pdf_options.do_ocr = True
pdf_options.do_table_structure = True
pdf_options.table_structure_options.do_cell_matching = True
pdf_options.generate_picture_images = True
pdf_options.images_scale = 1.0

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(


            # ---------------------------------------------
            # Give our OCR/table configuration to Docling.
            # ---------------------------------------------
            pipeline_options=pdf_options
        ),
})


def extract_doc(file_path: str):
    result = converter.convert(file_path)
    document = result.document

    markdown = document.export_to_markdown()
    txt = document.export_to_text()
    json_format = document.export_to_dict()

    Tables = []
    Pictures = []
    for item, _ in document.iterate_items():
        if isinstance(item, TableItem):
            Tables.append(item.export_to_dataframe(doc=document))
        elif isinstance(item, PictureItem):
            image = item.get_image(document)
            if image is not None:
                Pictures.append(image)

    return {
        "markdown": markdown,
        "txt": txt,
        "json": json_format,
        "tables": Tables,
        "pictures": Pictures,
    }


def save_results(result: dict, output_dir: str) -> None:
    import json
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "document.md").write_text(result["markdown"], encoding="utf-8")
    (out / "document.txt").write_text(result["txt"], encoding="utf-8")
    (out / "document.json").write_text(
        json.dumps(result["json"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    tables_dir = out / "tables"
    tables_dir.mkdir(exist_ok=True)
    for i, table in enumerate(result["tables"], start=1):
        table.to_csv(tables_dir / f"table_{i}.csv", index=False)

    pictures_dir = out / "pictures"
    pictures_dir.mkdir(exist_ok=True)
    for i, image in enumerate(result["pictures"], start=1):
        image.save(pictures_dir / f"picture_{i}.png")


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")

    pdf_path = Path(__file__).resolve().parent / "rn.pdf"
    output_dir = pdf_path.parent / f"{pdf_path.stem}_output"

    result = extract_doc(str(pdf_path))
    save_results(result, str(output_dir))

    print(
        f"Saved {len(result['tables'])} table(s), "
        f"{len(result['pictures'])} picture(s) to {output_dir}",
        flush=True,
    )

    # Docling's OCR/torch threads keep the process alive after completion.
    os._exit(0)

    