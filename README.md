import pandas as pd
from tqdm.notebook import tqdm
from google.colab import files
from bs4 import BeautifulSoup
import json
import re
import time
import requests

# ✅ STEP 1: Install Required Libraries
!pip install -q pandas tqdm beautifulsoup4 requests transformers

# ✅ STEP 2: Import Libraries
import pandas as pd
from tqdm.notebook import tqdm
from google.colab import files
from bs4 import BeautifulSoup
import json
import re
import time
import requests

# ✅ STEP 3: Setup DeepSeek API
deepseek_api_key = input("🔐 Enter your DeepSeek API Key: ")
deepseek_endpoint = "https://api.deepseek.ai/v1/rewrite"  # Replace with your actual endpoint

# ✅ STEP 4: Upload your CSV file
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
df = pd.read_csv(file_path)

# ✅ STEP 5: Setup Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# ✅ STEP 6: Clean JSON Extraction
def extract_json(text):
    try:
        clean = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
        clean = re.sub(r",\s*}", "}", clean)
        clean = re.sub(r",\s*]", "]", clean)
        match = re.search(r'\{[\s\S]*\}', clean)
        return json.loads(match.group()) if match else None
    except Exception as e:
        print("❌ JSON extraction failed:", e)
        return None

# ✅ STEP 7: Extract Rewriteable Text from HTML
def extract_rewriteable_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    texts = [tag.get_text() for tag in soup.find_all(['p', 'li']) if tag.get_text(strip=True)]
    return texts, soup

# ✅ STEP 8: Rewrite Paragraphs using DeepSeek API
def rewrite_html_texts(texts):
    if not texts:
        return []

    prompt = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    full_prompt = f"You are an SEO expert. Rewrite each line using long-tail SEO keywords. Only return the new versions.\n{prompt}"

    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "text": full_prompt,
        "model": "text-gen",  # Update this with the correct model name, if necessary
    }

    try:
        res = requests.post(deepseek_endpoint, headers=headers, json=data)
        res.raise_for_status()  # Raise an HTTPError for bad responses
        output = res.json()

        if 'error' in output:
            print(f"Error in API response: {output['error']}")
        else:
            lines = output['text']
            matches = re.findall(r"(\d+)\.\s*(.+)", lines)
            return [matches[i][1].strip() if i < len(matches) else texts[i] for i in range(len(texts))]
    except Exception as e:
        print(f"❌ DeepSeek rewriting failed: {e}")
        if 'res' in locals():
            print(f"API Response: {res.text}")  # Print the raw response text
        return texts

# ✅ STEP 9: Rebuild HTML with rewritten content
def rebuild_html(soup, new_texts):
    i = 0
    for tag in soup.find_all(['p', 'li']):
        if tag.get_text(strip=True) and i < len(new_texts):
            tag.string = new_texts[i]
            i += 1
    return str(soup)

# ✅ STEP 10: Rewrite Full Product
def rewrite_product(row):
    title = row.get('Title', 'This product')
    body_html = row.get("Body (HTML)", "")
    if isinstance(body_html, float) and pd.isna(body_html):
        body_html = ""

    if not body_html.strip():
        try:
            prompt = f"""Generate an SEO-optimized product description inside <p> or <ul><li> tags based only on this title:

Instructions:
- Make the paragraph section at least 3 full lines
- Add bullet points totaling around 300 words in <ul><li> format
- Use long-tail SEO keywords
- Relevant for Indian buyers
- Include specs, benefits, and use cases
- Return valid HTML using <p> and <ul><li> only
"""

            headers = {
                "Authorization": f"Bearer {deepseek_api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "text": prompt,
                "model": "text-gen",  # Update this with the correct model name, if necessary
            }

            res = requests.post(deepseek_endpoint, headers=headers, json=data)
            res.raise_for_status()  # Raise an HTTPError for bad responses
            output = res.json()
            body_html = output['text']
            body_html = re.sub(r"^(Here is[^<]*<)", "<", body_html)
            body_html = body_html.replace('Here is a product description that meets the requirements:', '').strip().replace('Here is a product description that meets the requirements:', '').strip()

        except Exception as e:
            print("❌ Error generating fallback Body (HTML):", e)
            body_html = f"<p>{title} is an imported product now available in India.</p>"

    text_list, soup = extract_rewriteable_text(body_html)
    rewritten_texts = rewrite_html_texts(text_list)
    rewritten_html = rebuild_html(soup, rewritten_texts)

    full_prompt = f"""
You are a product copywriter at Blumaple — a premium electronics brand in India that imports high-end office, consumer, Audio, Video, Gaming equipment, corporate gifting, Home improvement electronics, Personal accessories, photography, industrial tools, scientific tools, and industrial electronics from the USA. Our customers include IT companies, corporate buyers, upper-class and upper-middle-class professionals looking for advanced, hard-to-find products.

Rewrite the product data below to:
1. Create long-tail product title (max 120 characters)
2. Create Body (HTML) with SEO keywords
3. Create SEO title (max 80 characters)
4. Create meta description (140–160 characters)
5. Create 4 SEO bullet points
6. Google Shopping metadata
7. Image alt text
8. 5 Google search phrases

Input:
Title: {title}
Description: {rewritten_html}
Brand: {row.get('brand_name (product.metafields.custom.brand_name)', '')}
Bullets: {row.get('bullet point 1 (product.metafields.custom.bullet_point_1)', '')}, {row.get('bullet point 2 (product.metafields.custom.bullet_point_2)', '')}, {row.get('bullet point 3 (product.metafields.custom.bullet_point_3)', '')}, {row.get('bullet point 4 (product.metafields.custom.bullet_point_4)', '')}
Alt Text: {row.get('Image Alt Text', '')}
Boosts: {row.get('Search product boosts (product.metafields.shopify--discovery--product_search_boost.queries)', '')}

Return this JSON only:
{{
  "new_title": "...",
  "new_description": "...",
  "meta_description": "...",
  "brand_name": "...",
  "bullets": ["...", "...", "...", "..."],
  "category": "...",
  "gender": "...",
  "age_group": "...",
  "mpn": "...",
  "condition": "...",
  "custom_product": "...",
  "custom_labels": ["...", "...", "...", "...", "..."],
  "alt_text": "...",
  "search_boosts": ["...", "...", "...", "...", "..."]
}}
"""

    try:
        headers = {
            "Authorization": f"Bearer {deepseek_api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "text": full_prompt,
            "model": "text-gen",  # Update this with the correct model name, if necessary
        }

        res = requests.post(deepseek_endpoint, headers=headers, json=data)
        res.raise_for_status()  # Raise an HTTPError for bad responses
        output = res.json()
        response_text = output['text']
        result = extract_json(response_text)

        if result:
            result["new_description"] = rewritten_html
            result["fallback_html"] = body_html  # new
        return result
    except Exception as e:
        print("❌ Error on row:", e)
        if 'res' in locals():
            print(f"API Response: {res.text}")  # Print the raw response text
        return None

# ✅ STEP 10: Process All Rows
output_rows = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    result = rewrite_product(row)
    time.sleep(2)
    if result:
        row["Title"] = result.get("new_title", row["Title"])
        row["Body (HTML)"] = result.get("new_description", result.get("fallback_html", ""))
        row["SEO Title"] = result["new_title"]
        row["SEO Description"] = result["meta_description"]
        row["brand_name (product.metafields.custom.brand_name)"] = result["brand_name"]
        row["bullet point 1 (product.metafields.custom.bullet_point_1)"] = result["bullets"][0]
        row["bullet point 2 (product.metafields.custom.bullet_point_2)"] = result["bullets"][1]
        row["bullet point 3 (product.metafields.custom.bullet_point_3)"] = result["bullets"][2]
        row["bullet point 4 (product.metafields.custom.bullet_point_4)"] = result["bullets"][3]
        row["Google Shopping / Google Product Category"] = result["category"]
        row["Google Shopping / Gender"] = result["gender"]
        row["Google Shopping / Age Group"] = result["age_group"]
        row["Google Shopping / MPN"] = result["mpn"]
        row["Google Shopping / Condition"] = result["condition"]
        row["Google Shopping / Custom Product"] = result["custom_product"]

        labels = result.get("custom_labels", [])
        for j in range(5):
            row[f"Google Shopping / Custom Label {j}"] = labels[j] if j < len(labels) else ""

        row["Image Alt Text"] = result["alt_text"]
        row["Search product boosts (product.metafields.shopify--discovery--product_search_boost.queries)"] = ", ".join(result["search_boosts"])
    output_rows.append(row)

# ✅ STEP 11: Export Final Output
output_df = pd.DataFrame(output_rows)
output_df.to_csv("seo_products.csv", index=False)
files.download("seo_products.csv")
# SEO
