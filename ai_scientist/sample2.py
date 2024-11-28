import argparse
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import backoff
import requests
from llm import (AVAILABLE_LLMS, create_client, extract_json_between_markers,
                 get_response_from_llm)

S2_API_KEY = os.getenv("S2_API_KEY")


def generate_chapter_structure(theme, client, model) -> List[str]:
    # LLMに章立てを生成させる
    prompt = f"""You are an AI assistant tasked with creating a detailed chapter structure for a comprehensive review paper on the theme "{theme}".

Please provide a list of chapter titles that cover all the important aspects of this theme.

Exclude chapters on experiments and results.

Provide the chapter titles in the following JSON format:

<BEGIN OUTPUT>
{{
    "chapters": [
        "Introduction",
        "Background",
        ...
    ]
}}
<END OUTPUT>
"""

    response, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message="You are a helpful assistant that generates chapter structures for academic papers.",
        msg_history=[]
    )

    # JSON形式で章タイトルを抽出
    json_text = extract_text_between_markers(response, "<BEGIN OUTPUT>", "<END OUTPUT>")
    if json_text is None:
        print("Failed to extract chapter structure from LLM output.")
        return []

    try:
        chapters = json.loads(json_text).get("chapters", [])
    except json.JSONDecodeError:
        print("Failed to parse JSON from LLM output.")
        return []

    return chapters


def summarize_papers_for_chapter(chapter_title, theme, client, model) -> str:
    # 章に関連する論文を検索
    query = f"{theme} {chapter_title}"
    papers = search_for_papers(query, result_limit=10)
    if not papers:
        print(f"No relevant papers were found for chapter: {chapter_title}")
        return ""

    # 論文情報を文字列にまとめる
    papers_text = ""
    for i, paper in enumerate(papers):
        authors = ', '.join([author['name'] for author in paper['authors']])
        papers_text += f"{i+1}. {paper['title']} by {authors} ({paper['year']})\n"
        papers_text += f"Abstract: {paper['abstract']}\n\n"

    # LLMを使って論文をまとめる
    summary_prompt = f"""You are an AI language model assistant tasked with summarizing research papers in detail.

Please provide a detailed summary in English of the following papers related to the chapter "{chapter_title}" in the context of "{theme}".

List each paper with its key contributions, methods, and findings.

Here are the papers:

{papers_text}
"""

    summary, _ = get_response_from_llm(
        summary_prompt,
        client=client,
        model=model,
        system_message="You are a knowledgeable researcher. Summarize the following papers in detail.",
        msg_history=[]
    )

    return summary


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Optional[List[Dict]]:
    if not query:
        return None
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers


def extract_text_between_markers(text, start_marker, end_marker):
    pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return None


def generate_latex_review_paper(theme, chapters_summaries, client, model, output_file="generated_review_paper.tex"):
    # LaTeX生成のためのプロンプトを作成
    prompt = f"""You are an AI assistant tasked with writing a comprehensive review paper in LaTeX format based on the provided summaries for each chapter.

Theme: "{theme}"

Please write a full review paper in LaTeX format, including necessary sections such as Abstract, Introduction, Background, and Conclusion. Exclude sections on Experiments and Results.

Use the provided summaries for each chapter to construct the content.

Provide the LaTeX code between <BEGIN LATEX> and <END LATEX> markers.

Here are the chapter summaries:

"""

    for chapter_title, summary in chapters_summaries.items():
        prompt += f"### {chapter_title}\n{summary}\n\n"

    # LLMを使ってLaTeXコードを生成
    response, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message="You are a helpful assistant that writes comprehensive LaTeX review papers.",
        msg_history=[]
    )

    # マーカー間のLaTeXコードを抽出
    latex_code = extract_text_between_markers(response, "<BEGIN LATEX>", "<END LATEX>")

    if latex_code is None:
        print("Failed to extract LaTeX code from LLM output.")
        return

    # LaTeXコードをファイルに書き込む
    with open(output_file, "w") as f:
        f.write(latex_code)

    print(f"LaTeX file has been written to {output_file}.")


def compile_latex_file(tex_file, output_pdf="output.pdf", timeout=30):
    cwd = os.path.dirname(os.path.abspath(tex_file))
    tex_filename = os.path.basename(tex_file)
    commands = [
        ["pdflatex", "-interaction=nonstopmode", tex_filename],
        ["bibtex", tex_filename.replace(".tex", "")],
        ["pdflatex", "-interaction=nonstopmode", tex_filename],
        ["pdflatex", "-interaction=nonstopmode", tex_filename],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)
        except subprocess.TimeoutExpired:
            print(f"Latex timed out after {timeout} seconds")
            return
        except subprocess.CalledProcessError as e:
            print(f"Error running command {' '.join(command)}: {e}")
            return

    # PDFファイルを出力
    generated_pdf = os.path.join(cwd, tex_filename.replace(".tex", ".pdf"))
    if os.path.exists(generated_pdf):
        shutil.move(generated_pdf, output_pdf)
        print(f"PDF has been generated as {output_pdf}.")
    else:
        print("Failed to generate PDF.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="テーマに関連する論文を検索し、要約し、レビュー論文を生成します。")
    parser.add_argument(
        "--theme",
        type=str,
        required=True,
        help="検索したいテーマを指定してください。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="使用する LLM のモデルを指定してください。",
    )
    args = parser.parse_args()

    # クライアントを作成
    client, client_model = create_client(args.model)

    # 章立てを生成
    chapters = generate_chapter_structure(
        theme=args.theme,
        client=client,
        model=client_model,
    )

    if not chapters:
        print("No chapters were generated.")
        exit()

    # 各章について論文を検索し、要約を生成
    chapters_summaries = {}
    for chapter_title in chapters:
        print(f"Processing chapter: {chapter_title}")
        summary = summarize_papers_for_chapter(
            chapter_title=chapter_title,
            theme=args.theme,
            client=client,
            model=client_model,
        )
        chapters_summaries[chapter_title] = summary

    # 要約をもとにLaTeXレビュー論文を生成
    generate_latex_review_paper(
        theme=args.theme,
        chapters_summaries=chapters_summaries,
        client=client,
        model=client_model,
        output_file="generated_review_paper.tex"
    )

    # LaTeXファイルをコンパイルしてPDFを生成
    compile_latex_file("generated_review_paper.tex", output_pdf="generated_review_paper.pdf")
