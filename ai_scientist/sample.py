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


def summarize_papers_on_theme(theme, client, model) -> str:
    # 論文を検索
    papers = search_for_papers(theme, result_limit=10)
    if not papers:
        print("No relevant papers were found.")
        return ""

    # 論文情報を文字列にまとめる
    papers_text = ""
    for i, paper in enumerate(papers):
        authors = ', '.join([author['name'] for author in paper['authors']])
        papers_text += f"{i+1}. {paper['title']} by {authors} ({paper['year']})\n"
        papers_text += f"Abstract: {paper['abstract']}\n\n"

    # LLM を使って論文をまとめる
    summary_prompt = f"""You are an AI language model assistant tasked with summarizing research papers in detail.

Please provide a detailed summary in English of the following papers related to the theme "{theme}".

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


def generate_latex_from_summary(summary, client, model, output_file="generated_paper.tex"):
    # LaTeX生成のためのプロンプトを作成
    prompt = f"""You are an AI assistant tasked with writing a research paper in LaTeX format based on the following summary of research papers:

{summary}

Please write a full research paper in LaTeX format, including all necessary sections such as Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion, and References.

Ensure that the paper is well-structured and the LaTeX code is properly formatted.

Provide the LaTeX code between <BEGIN LATEX> and <END LATEX> markers."""

    # LLMを使ってLaTeXコードを生成
    response, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message="You are a helpful assistant that writes LaTeX research papers.",
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

    parser = argparse.ArgumentParser(description="テーマに関連する論文を検索し、要約し、論文を生成します。")
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

    # テーマに関連する論文を検索し、要約
    summary = summarize_papers_on_theme(
        theme=args.theme,
        client=client,
        model=client_model,
    )

    if summary:
        # 要約をもとにLaTeX論文を生成
        generate_latex_from_summary(
            summary=summary,
            client=client,
            model=client_model,
            output_file="generated_paper.tex"
        )

        # LaTeXファイルをコンパイルしてPDFを生成
        compile_latex_file("generated_paper.tex", output_pdf="generated_paper.pdf")
    else:
        print("No summary was generated.")
