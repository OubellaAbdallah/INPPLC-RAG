"""Table and image captioning for INPPLC knowledge base.

Generates detailed captions and summaries for RAG retrieval using
the Mistral chat API.
"""

from __future__ import annotations

import time
from typing import Any

from mistralai.client import Mistral

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

TABLE_SUMMARIZER_PROMPT = """You are an expert data analyst specializing in governance and anti-corruption information.
Your task is to create a detailed and comprehensive summary of the following table
from an INPPLC (Instance Nationale de la Probité, de la Prévention et de la Lutte contre la Corruption) document.

CRITICAL REQUIREMENTS:
1. Extract ALL numerical data - every number, percentage, value, and measure present
2. Preserve the EXACT values - do not round or approximate
3. Include ALL row and column headers with their complete labels
4. Capture every footnote and annotation
5. Note the units of measurement for every variable
6. Include sample sizes for every group/subgroup
7. Document any missing data indicators

DETAILED ANALYSIS REQUIREMENTS:
- Identify the data structure and relationships shown
- Note any statistical measures or values presented
- Describe any patterns or trends in the data
- Include any contextual information about the data source
- Note relevant governance or anti-corruption metrics

CONTENT CONTEXT:
- Identify topics or domains covered (corruption indices, governance indicators, etc.)
- Note any demographic or geographic information presented
- Document any institutional or regulatory measures
- Capture quality scores or performance indicators if present
- Include any outcome or impact metrics

METHODOLOGY DETAILS:
- Sample recruitment method and eligibility criteria reflected in the table
- Any adjustment variables in analyses
- Model specifications if applicable
- Reference groups used for comparisons

Table content:
{data}

Additional context from metadata:
{metadata}

Provide an exhaustive summary that captures EVERY piece of information in this table.
A reader should be able to reconstruct the complete table from your summary alone.
Begin with a one-sentence overview of what the table presents, then systematically describe
every section, column, and row with all associated values.
Use the document's original language (French, Arabic, or English)."""

IMAGE_SUMMARIZER_PROMPT = """You are an expert data visualization analyst.
Your task is to provide a detailed description and analysis of the following figure
from an INPPLC (Instance Nationale de la Probité, de la Prévention et de la Lutte contre la Corruption) document.

FIGURE TYPE IDENTIFICATION:
First, identify the type of figure:
- Bar chart / Grouped bar chart
- Line graph / Time series
- Scatter plot / Correlation plot
- Box plot / Violin plot
- Heat map / Correlation matrix
- Flow diagram / Organizational chart
- Process diagram / Schema
- Histogram
- Pie chart
- Other (specify)

AXES AND SCALES:
- X-axis: label, units, range, categories
- Y-axis: label, units, range, scale type
- Any axis labels and tick marks

LEGEND AND ANNOTATIONS:
- All legend entries with their visual markers
- Text annotations, arrows, or callouts
- Statistical significance markers
- Reference lines or thresholds

QUANTITATIVE DATA EXTRACTION:
- Extract values visible in the figure
- Note peaks, troughs, and trends
- Identify patterns and relationships
- Note any outliers

STATISTICAL INDICATORS:
- P-values or significance markers
- Confidence intervals or bands
- Correlation coefficients
- Sample sizes

TEMPORAL OR CATEGORICAL ANALYSIS:
- Time points or categories shown
- Patterns across groups or time
- Rate of change where applicable

DOMAIN-SPECIFIC INSIGHTS:
- Key governance or anti-corruption metrics shown
- Trends or changes depicted
- Comparisons between entities, regions, or time periods

Figure metadata:
{metadata}

Provide a comprehensive description that allows full understanding of the figure's content,
trends, patterns, and key findings. Include numerical values where possible.
Use the document's original language (French, Arabic, or English)."""

CHAT_MODEL = "mistral-large-latest"


class TableCaptioner:
    """Generates detailed summaries of tables using Mistral chat API.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        api_key = self._settings.mistral_api_key
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set. Add it to your .env file."
            )
        self._client = Mistral(api_key=api_key)
        self._last_call_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_call_time = time.time()

    def generate_summary(
        self,
        table_content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate a detailed natural language summary of a table.

        Args:
            table_content: HTML or markdown table content.
            metadata: Additional context about the table's source.

        Returns:
            Detailed summary string.
        """
        self._rate_limit()
        metadata_str = str(metadata) if metadata else "No additional metadata"
        prompt = TABLE_SUMMARIZER_PROMPT.format(
            data=table_content,
            metadata=metadata_str,
        )

        try:
            response = self._client.chat.complete(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("table_summary_error", error=str(e))
            return f"[Table summary unavailable: {e}]"

    def summarize_tables(
        self,
        table_elements: list[dict[str, Any]],
    ) -> list[str]:
        """Summarize a batch of table elements.

        Args:
            table_elements: List of dicts with 'content' and 'metadata' keys.

        Returns:
            List of summary strings, one per table.
        """
        summaries: list[str] = []

        for i, table in enumerate(table_elements):
            logger.info(
                "summarizing_table",
                index=i + 1,
                total=len(table_elements),
            )
            summary = self.generate_summary(
                table["content"],
                table.get("metadata"),
            )
            summaries.append(summary)

        logger.info("table_summarization_complete", count=len(summaries))
        return summaries


class ImageCaptioner:
    """Generates detailed captions of images using Mistral chat API.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        api_key = self._settings.mistral_api_key
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set. Add it to your .env file."
            )
        self._client = Mistral(api_key=api_key)
        self._last_call_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_call_time = time.time()

    def generate_caption(
        self,
        image_base64: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate a detailed caption for an image.

        Args:
            image_base64: Base64-encoded image string.
            metadata: Additional context about the image's source.

        Returns:
            Detailed caption string.
        """
        self._rate_limit()
        metadata_str = str(metadata) if metadata else "No additional metadata"
        prompt_text = IMAGE_SUMMARIZER_PROMPT.format(metadata=metadata_str)

        try:
            response = self._client.chat.complete(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{image_base64}",
                            },
                        ],
                    }
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("image_caption_error", error=str(e))
            return f"[Image caption unavailable: {e}]"

    def caption_images(
        self,
        image_elements: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        """Caption a batch of image elements.

        Args:
            image_elements: List of dicts with 'content' (base64) and 'metadata' keys.

        Returns:
            Tuple of (captions, base64_strings).
        """
        captions: list[str] = []
        base64_list: list[str] = []

        for i, image in enumerate(image_elements):
            logger.info(
                "captioning_image",
                index=i + 1,
                total=len(image_elements),
            )
            caption = self.generate_caption(
                image["content"],
                image.get("metadata"),
            )
            captions.append(caption)
            base64_list.append(image["content"])

        logger.info("image_captioning_complete", count=len(captions))
        return captions, base64_list
