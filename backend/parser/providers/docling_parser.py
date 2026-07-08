"""Docling-based implementation of the `DocumentParser` port.

This is the only file in the application permitted to import `docling` or
`docling_core`. Every Docling type is translated into the Docling-agnostic
`ExtractedDocument` contract before this module returns -- nothing Docling
-shaped survives past `parse()`.
"""

import io
import logging

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

from backend.parser.exceptions import UnreadablePdfError
from backend.parser.interfaces.document_parser import DocumentParser
from backend.parser.interfaces.extracted_document import (
    ExtractedBoundingBox,
    ExtractedContentItem,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    ExtractedTableCell,
    ExtractedTextBlock,
    ExtractedTextRole,
)

logger = logging.getLogger(__name__)

_EXCLUDED_LABELS = frozenset(
    {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.CAPTION, DocItemLabel.TITLE}
)


class DoclingDocumentParser(DocumentParser):
    """Parses PDFs into `ExtractedDocument` using IBM Docling.

    Attributes:
        ocr_enabled: Whether OCR is run for scanned/image-based pages.
    """

    def __init__(self, ocr_enabled: bool = False, image_scale: float = 2.0) -> None:
        """Initialize the parser and its underlying Docling pipeline.

        Constructing this class loads Docling's layout and table-structure
        models, which is expensive -- callers should construct one instance
        and reuse it across requests rather than creating one per parse.

        Args:
            ocr_enabled: Whether to run OCR for scanned/image-based pages.
                Docling has no first-party PaddleOCR backend; enabling this
                uses whichever OCR engine Docling selects by default.
            image_scale: Resolution multiplier used when rendering extracted
                figure images.
        """
        self.ocr_enabled = ocr_enabled
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr_enabled
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = image_scale
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def parse(self, pdf_bytes: bytes) -> ExtractedDocument:
        docling_document = self._convert(pdf_bytes)
        title = self._extract_title(docling_document)
        content = self._extract_content(docling_document, title_text=title)
        return ExtractedDocument(
            title=title,
            page_count=len(docling_document.pages),
            content=tuple(content),
        )

    def _convert(self, pdf_bytes: bytes) -> DoclingDocument:
        stream = DocumentStream(name="document.pdf", stream=io.BytesIO(pdf_bytes))
        try:
            result = self._converter.convert(stream)
        except ConversionError as exc:
            raise UnreadablePdfError(reason=str(exc)) from exc
        return result.document

    def _extract_title(self, doc: DoclingDocument) -> str | None:
        """Return the document title, falling back to the first heading.

        Docling's `TITLE` label is not reliably assigned -- on some
        documents the title is only ever labeled as a `SECTION_HEADER`.
        """
        for item, _level in doc.iterate_items():
            if isinstance(item, TextItem) and item.label == DocItemLabel.TITLE:
                return item.text.strip() or None
        for item, _level in doc.iterate_items():
            if isinstance(item, SectionHeaderItem):
                return item.text.strip() or None
        return None

    def _extract_content(
        self, doc: DoclingDocument, *, title_text: str | None
    ) -> list[ExtractedContentItem]:
        """Walk the document in reading order, building the content stream.

        The item used as the title fallback (a section header) is excluded
        here so it is not also emitted as a section.
        """
        title_already_consumed = False
        content: list[ExtractedContentItem] = []

        for item, _level in doc.iterate_items():
            label = getattr(item, "label", None)

            if label == DocItemLabel.TABLE and isinstance(item, TableItem):
                content.append(self._extract_table(item, doc))
                continue

            if label == DocItemLabel.PICTURE and isinstance(item, PictureItem):
                content.append(self._extract_figure(item, doc))
                continue

            if label in _EXCLUDED_LABELS:
                continue

            if not isinstance(item, TextItem):
                continue

            if (
                isinstance(item, SectionHeaderItem)
                and not title_already_consumed
                and item.text.strip() == title_text
            ):
                title_already_consumed = True
                continue

            content.append(self._extract_text_block(item, doc))

        return content

    def _extract_text_block(self, item: TextItem, doc: DoclingDocument) -> ExtractedTextBlock:
        role = (
            ExtractedTextRole.SECTION_HEADER
            if isinstance(item, SectionHeaderItem)
            else ExtractedTextRole.PARAGRAPH
        )
        level = item.level if isinstance(item, SectionHeaderItem) else None
        return ExtractedTextBlock(
            role=role,
            text=item.text,
            level=level,
            bounding_boxes=self._bounding_boxes(item, doc),
        )

    def _extract_table(self, item: TableItem, doc: DoclingDocument) -> ExtractedTable:
        cells = tuple(
            ExtractedTableCell(
                row=cell.start_row_offset_idx,
                column=cell.start_col_offset_idx,
                text=cell.text,
                row_span=cell.row_span,
                column_span=cell.col_span,
                is_header=cell.column_header or cell.row_header,
            )
            for cell in item.data.table_cells
        )
        markdown = self._safe_export_markdown(item, doc)
        caption_text, caption_boxes = self._resolve_caption(item, doc)
        return ExtractedTable(
            cells=cells,
            num_rows=item.data.num_rows,
            num_columns=item.data.num_cols,
            markdown=markdown,
            caption_text=caption_text,
            caption_bounding_boxes=caption_boxes,
            bounding_boxes=self._bounding_boxes(item, doc),
        )

    def _extract_figure(self, item: PictureItem, doc: DoclingDocument) -> ExtractedFigure:
        image_bytes: bytes | None = None
        image_format: str | None = None
        image = item.get_image(doc)
        if image is not None:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            image_format = "png"

        caption_text, caption_boxes = self._resolve_caption(item, doc)
        return ExtractedFigure(
            caption_text=caption_text,
            caption_bounding_boxes=caption_boxes,
            image_bytes=image_bytes,
            image_format=image_format,
            bounding_boxes=self._bounding_boxes(item, doc),
        )

    def _resolve_caption(
        self, item: TableItem | PictureItem, doc: DoclingDocument
    ) -> tuple[str | None, tuple[ExtractedBoundingBox, ...]]:
        if not item.captions:
            return None, ()
        caption_item = item.captions[0].resolve(doc)
        return caption_item.text, self._bounding_boxes(caption_item, doc)

    def _safe_export_markdown(self, item: TableItem, doc: DoclingDocument) -> str | None:
        try:
            return item.export_to_markdown(doc=doc)
        except Exception:
            logger.warning("failed to export table to markdown", exc_info=True)
            return None

    def _bounding_boxes(
        self, item: TextItem | TableItem | PictureItem, doc: DoclingDocument
    ) -> tuple[ExtractedBoundingBox, ...]:
        boxes = []
        for prov in item.prov:
            page = doc.pages.get(prov.page_no)
            bbox = prov.bbox
            if page is not None:
                bbox = bbox.to_top_left_origin(page.size.height)
            boxes.append(
                ExtractedBoundingBox(
                    page_number=prov.page_no, x0=bbox.l, y0=bbox.t, x1=bbox.r, y1=bbox.b
                )
            )
        return tuple(boxes)
