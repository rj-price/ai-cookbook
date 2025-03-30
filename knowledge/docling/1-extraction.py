from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls

converter = DocumentConverter()

# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------

#result = converter.convert("https://arxiv.org/pdf/2408.09869")
#
#document = result.document
#markdown_output = document.export_to_markdown()
#json_output = document.export_to_dict()
#
#print(markdown_output)

# --------------------------------------------------------------
# Basic HTML extraction
# --------------------------------------------------------------

#result = converter.convert("https://ds4sd.github.io/docling/")
#result = converter.convert("https://www.analyticsvidhya.com/blog/2025/03/embedding-for-rag-models/")
#
#document = result.document
#markdown_output = document.export_to_markdown()
#print(markdown_output)

# --------------------------------------------------------------
# Scrape multiple pages using the sitemap
# --------------------------------------------------------------

#sitemap_urls = get_sitemap_urls("https://ds4sd.github.io/docling/")
sitemap_urls = get_sitemap_urls("https://www.analyticsvidhya.com/blog/2025/03/")
conv_results_iter = converter.convert_all(sitemap_urls)

docs = []
for result in conv_results_iter:
    if result.document:
        document = result.document
        docs.append(document)

print(docs[0].export_to_markdown())