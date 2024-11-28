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


def collect_papers_on_theme(theme) -> List[Dict]:
    # 50件の論文を収集
    papers_collected = []
    seen_paper_ids = set()
    for offset in range(0, 50, 10):
        papers = search_for_papers(theme, result_limit=10, offset=offset)
        if not papers:
            print("No more relevant papers were found.")
            break
        for paper in papers:
            paper_id = paper.get('paperId')
            if paper_id not in seen_paper_ids:
                papers_collected.append(paper)
                seen_paper_ids.add(paper_id)
    return papers_collected


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10, offset=0) -> Optional[List[Dict]]:
    if not query:
        return None
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "offset": offset,
            "fields": "paperId,title,authors,venue,year,abstract,citationStyles,citationCount",
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


def get_bibtex_entry(paperId) -> Optional[str]:
    if not paperId:
        return None
    rsp = requests.get(
        f"https://api.semanticscholar.org/{paperId}?format=bibtex",
        headers={"X-API-KEY": S2_API_KEY},
    )
    print(f"Fetching BibTeX for paperId {paperId}: Status Code {rsp.status_code}")
    rsp.raise_for_status()
    bibtex_entry = rsp.text.strip()
    return bibtex_entry


def extract_text_between_markers(text, start_marker, end_marker):
    pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return None


def generate_review_paper_from_papers(papers_text, theme, bibtex_entries, client, model, output_file="generated_paper.tex"):
    # レビュー論文生成のためのプロンプトを作成

    # BibTeXエントリを一つの文字列にまとめる
    bibtex_content = '\n\n'.join(bibtex_entries)

    prompt = f"""You are an AI assistant tasked with writing a detailed review paper in LaTeX format on the theme "{theme}".

Based on the following list of papers, write a comprehensive review that includes critical analysis and synthesis of the topics covered. The review should be well-structured and cover the main ideas and contributions of the papers.

Include a reference list at the end, properly formatted in LaTeX. The references are provided in the 'references.bib' file.

Do not include sections such as Experiments, Results, or Related Work, as this is a review paper.

Here are the papers:

{papers_text}

Below is the content of the 'references.bib' file:

<BEGIN BIBTEX>
{bibtex_content}
<END BIBTEX>

Please write the full review paper in LaTeX format, including all necessary sections such as Abstract, Introduction, Main Body, Conclusion, and References.

Ensure that the paper is well-structured and the LaTeX code is properly formatted.

Include the 'references.bib' content in the LaTeX code using the following format:

\\begin{{filecontents}}{{references.bib}}
{bibtex_content}
\\end{{filecontents}}

Provide the LaTeX code between <BEGIN LATEX> and <END LATEX> markers."""

    # LLMを使ってLaTeXコードを生成
    response, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message="You are a helpful assistant that writes LaTeX review papers.",
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

    parser = argparse.ArgumentParser(description="テーマに関連する論文を検索し、レビュー論文を生成します。")
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

    # テーマに関連する論文を検索し、情報を収集
    papers = collect_papers_on_theme(
        theme=args.theme,
    )

    if papers:
        # 論文情報を文字列にまとめ、BibTeXエントリを取得
        papers_text = ""
        bibtex_entries = []
        for i, paper in enumerate(papers):
            authors = ', '.join([author['name'] for author in paper['authors']])
            papers_text += f"{i+1}. {paper['title']} by {authors} ({paper['year']})\n"
            papers_text += f"Abstract: {paper['abstract']}\n\n"

            # BibTeXエントリを取得
            paperId = paper.get('paperId')
            bibtex_entry = get_bibtex_entry(paperId)
            if bibtex_entry:
                bibtex_entries.append(bibtex_entry)
            else:
                print(f"Could not retrieve BibTeX entry for paperId {paperId}")

        # BibTeXエントリを 'references.bib' に書き出し
        with open('references.bib', 'w') as bibfile:
            bibfile.write('\n\n'.join(bibtex_entries))

        # 論文情報をもとにレビュー論文を生成
        generate_review_paper_from_papers(
            papers_text=papers_text,
            theme=args.theme,
            bibtex_entries=bibtex_entries,
            client=client,
            model=client_model,
            output_file="generated_paper.tex"
        )

        # LaTeXファイルをコンパイルしてPDFを生成
        compile_latex_file("generated_paper.tex", output_pdf="generated_paper.pdf")
    else:
        print("No papers were collected.")



# import argparse
# import json
# import os
# import os.path as osp
# import re
# import shutil
# import subprocess
# import time
# from typing import Dict, List, Union

# import requests
# from llm import (AVAILABLE_LLMS, create_client, extract_json_between_markers,
#                  get_response_from_llm)

# # LLMとのインタラクション用の関数（get_response_from_llm）は、適切なライブラリやAPIを使用して実装してください。

# S2_API_KEY = os.getenv("S2_API_KEY")


# def extract_text_between_markers(text, start_marker, end_marker):
#     pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), re.DOTALL)
#     match = pattern.search(text)
#     if match:
#         return match.group(1).strip()
#     else:
#         return None

# def search_for_papers(query: str, result_limit: int = 10) -> Union[None, List[Dict]]:
#     if not query:
#         return None
#     rsp = requests.get(
#         "https://api.semanticscholar.org/graph/v1/paper/search",
#         headers={"X-API-KEY": S2_API_KEY},
#         params={
#             "query": query,
#             "limit": result_limit,
#             "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
#         },
#     )
#     time.sleep(1.0)
#     if rsp.status_code != 200:
#         print(f"Error: {rsp.status_code}")
#         return None
#     results = rsp.json()
#     total = results.get("total", 0)
#     if total == 0:
#         return None
#     papers = results["data"]
#     return papers


# def collect_papers_on_theme(theme, client, model) -> List[Dict]:
#     collected_papers = []
#     retrieved_titles = set()  # 重複防止のためのセット
#     attempts = 0

#     while len(collected_papers) < 50 and attempts < 10:
#         papers = search_for_papers(theme, result_limit=10)
#         if not papers:
#             print("No relevant papers were found in this attempt.")
#             attempts += 1
#             continue

#         for paper in papers:
#             if paper['title'] not in retrieved_titles:
#                 collected_papers.append(paper)
#                 retrieved_titles.add(paper['title'])

#             if len(collected_papers) >= 50:
#                 break

#         attempts += 1

#     if len(collected_papers) < 50:
#         print(f"Warning: Only collected {len(collected_papers)} papers after {attempts} attempts.")

#     return collected_papers


# def generate_review_paper(papers: List[Dict], client, model, theme: str, output_file="review_paper.tex"):
#     # 論文情報を文字列にまとめる
#     papers_text = ""
#     for i, paper in enumerate(papers):
#         authors = ', '.join([author['name'] for author in paper['authors']])
#         papers_text += f"{i+1}. {paper['title']} by {authors} ({paper['year']})\n"
#         papers_text += f"Abstract: {paper['abstract']}\n\n"

#     # レビュー論文生成のためのプロンプトを作成
#     review_prompt = f"""You are an AI assistant tasked with writing a detailed review paper in LaTeX format.

# Please write a comprehensive review paper in English on the theme "{theme}" based on the following 50 research papers.

# Focus on synthesizing the state of the art, key contributions, methods, and gaps in the research. 
# Avoid sections like Experiments, Results, or Related Work.

# Here are the papers:

# {papers_text}

# Include proper citations for each paper. Provide the LaTeX code between <BEGIN LATEX> and <END LATEX> markers."""

#     # LLMを使ってLaTeXコードを生成
#     response, _ = get_response_from_llm(
#         review_prompt,
#         client=client,
#         model=model,
#         system_message="You are a helpful assistant writing LaTeX review papers.",
#         msg_history=[]
#     )

#     # マーカー間のLaTeXコードを抽出
#     latex_code = extract_text_between_markers(response, "<BEGIN LATEX>", "<END LATEX>")

#     if latex_code is None:
#         print("Failed to extract LaTeX code from LLM output.")
#         return

#     # LaTeXコードをファイルに書き込む
#     with open(output_file, "w") as f:
#         f.write(latex_code)

#     print(f"LaTeX review paper has been written to {output_file}.")


# def compile_latex_file(tex_file, output_pdf="output.pdf", timeout=30):
#     cwd = os.path.dirname(os.path.abspath(tex_file))
#     tex_filename = os.path.basename(tex_file)
#     commands = [
#         ["pdflatex", "-interaction=nonstopmode", tex_filename],
#         ["bibtex", tex_filename.replace(".tex", "")],
#         ["pdflatex", "-interaction=nonstopmode", tex_filename],
#         ["pdflatex", "-interaction=nonstopmode", tex_filename],
#     ]

#     for command in commands:
#         try:
#             result = subprocess.run(
#                 command,
#                 cwd=cwd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 timeout=timeout,
#             )
#             print("Standard Output:\n", result.stdout)
#             print("Standard Error:\n", result.stderr)
#         except subprocess.TimeoutExpired:
#             print(f"Latex timed out after {timeout} seconds")
#             return
#         except subprocess.CalledProcessError as e:
#             print(f"Error running command {' '.join(command)}: {e}")
#             return

#     # PDFファイルを出力
#     generated_pdf = os.path.join(cwd, tex_filename.replace(".tex", ".pdf"))
#     if os.path.exists(generated_pdf):
#         shutil.move(generated_pdf, output_pdf)
#         print(f"PDF has been generated as {output_pdf}.")
#     else:
#         print("Failed to generate PDF.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="テーマに関連する論文を収集し、レビュー論文を生成します。")
#     parser.add_argument(
#         "--theme",
#         type=str,
#         required=True,
#         help="検索したいテーマを指定してください。",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="gpt-4o-2024-05-13",
#         choices=AVAILABLE_LLMS,
#         help="使用する LLM のモデルを指定してください。",
#     )
#     args = parser.parse_args()

#     # クライアントを作成
#     client, client_model = create_client(args.model)

#     # テーマに関連する論文を収集
#     papers = collect_papers_on_theme(theme=args.theme, client=client, model=client_model)

#     if papers:
#         # レビュー論文を生成
#         generate_review_paper(
#             papers=papers,
#             client=client,
#             model=client_model,
#             theme=args.theme,
#             output_file="review_paper.tex"
#         )

#         # LaTeXファイルをコンパイルしてPDFを生成
#         compile_latex_file("review_paper.tex", output_pdf="review_paper.pdf")
#     else:
#         print("No papers were collected.")

# import argparse
# import json
# import os
# import os.path as osp
# import re
# import shutil
# import subprocess
# import time
# from typing import Dict, List, Optional, Tuple

# import backoff
# import requests
# from llm import (AVAILABLE_LLMS, create_client, extract_json_between_markers,
#                  get_response_from_llm)

# S2_API_KEY = os.getenv("S2_API_KEY")


# def collect_papers_on_theme(theme) -> List[Dict]:
#     # 50件の論文を収集
#     papers_collected = []
#     seen_paper_ids = set()
#     for offset in range(0, 50, 10):
#         papers = search_for_papers(theme, result_limit=10, offset=offset)
#         if not papers:
#             print("No more relevant papers were found.")
#             break
#         for paper in papers:
#             paper_id = paper.get('paperId')
#             if paper_id not in seen_paper_ids:
#                 papers_collected.append(paper)
#                 seen_paper_ids.add(paper_id)
#     return papers_collected


# def on_backoff(details):
#     print(
#         f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
#         f"calling function {details['target'].__name__} at {time.strftime('%X')}"
#     )


# @backoff.on_exception(
#     backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
# )
# def search_for_papers(query, result_limit=10, offset=0) -> Optional[List[Dict]]:
#     if not query:
#         return None
#     rsp = requests.get(
#         "https://api.semanticscholar.org/graph/v1/paper/search",
#         headers={"X-API-KEY": S2_API_KEY},
#         params={
#             "query": query,
#             "limit": result_limit,
#             "offset": offset,
#             "fields": "paperId,title,authors,venue,year,abstract,citationStyles,citationCount",
#         },
#     )
#     print(f"Response Status Code: {rsp.status_code}")
#     print(
#         f"Response Content: {rsp.text[:500]}"
#     )  # Print the first 500 characters of the response content
#     rsp.raise_for_status()
#     results = rsp.json()
#     total = results["total"]
#     time.sleep(1.0)
#     if not total:
#         return None

#     papers = results["data"]
#     return papers


# def extract_text_between_markers(text, start_marker, end_marker):
#     pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), re.DOTALL)
#     match = pattern.search(text)
#     if match:
#         return match.group(1).strip()
#     else:
#         return None


# def generate_review_paper_from_papers(papers_text, theme, client, model, output_file="generated_paper.tex"):
#     # レビュー論文生成のためのプロンプトを作成
#     prompt = f"""You are an AI assistant tasked with writing a detailed review paper in LaTeX format on the theme "{theme}".

# Based on the following list of papers, write a comprehensive review that includes critical analysis and synthesis of the topics covered. The review should be well-structured and cover the main ideas and contributions of the papers.

# Include a reference list at the end, properly formatted in LaTeX.

# Do not include sections such as Experiments, Results, or Related Work, as this is a review paper.

# Here are the papers:

# {papers_text}

# Please write the full review paper in LaTeX format, including all necessary sections such as Abstract, Introduction, Main Body, Conclusion, and References.

# Ensure that the paper is well-structured and the LaTeX code is properly formatted.

# Provide the LaTeX code between <BEGIN LATEX> and <END LATEX> markers."""

#     # LLMを使ってLaTeXコードを生成
#     response, _ = get_response_from_llm(
#         prompt,
#         client=client,
#         model=model,
#         system_message="You are a helpful assistant that writes LaTeX review papers.",
#         msg_history=[]
#     )

#     # マーカー間のLaTeXコードを抽出
#     latex_code = extract_text_between_markers(response, "<BEGIN LATEX>", "<END LATEX>")

#     if latex_code is None:
#         print("Failed to extract LaTeX code from LLM output.")
#         return

#     # LaTeXコードをファイルに書き込む
#     with open(output_file, "w") as f:
#         f.write(latex_code)

#     print(f"LaTeX file has been written to {output_file}.")


# def compile_latex_file(tex_file, output_pdf="output.pdf", timeout=30):
#     cwd = os.path.dirname(os.path.abspath(tex_file))
#     tex_filename = os.path.basename(tex_file)
#     commands = [
#         ["pdflatex", "-interaction=nonstopmode", tex_filename],
#         ["bibtex", tex_filename.replace(".tex", "")],
#         ["pdflatex", "-interaction=nonstopmode", tex_filename],
#         ["pdflatex", "-interaction=nonstopmode", tex_filename],
#     ]

#     for command in commands:
#         try:
#             result = subprocess.run(
#                 command,
#                 cwd=cwd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 timeout=timeout,
#             )
#             print("Standard Output:\n", result.stdout)
#             print("Standard Error:\n", result.stderr)
#         except subprocess.TimeoutExpired:
#             print(f"Latex timed out after {timeout} seconds")
#             return
#         except subprocess.CalledProcessError as e:
#             print(f"Error running command {' '.join(command)}: {e}")
#             return

#     # PDFファイルを出力
#     generated_pdf = os.path.join(cwd, tex_filename.replace(".tex", ".pdf"))
#     if os.path.exists(generated_pdf):
#         shutil.move(generated_pdf, output_pdf)
#         print(f"PDF has been generated as {output_pdf}.")
#     else:
#         print("Failed to generate PDF.")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="テーマに関連する論文を検索し、レビュー論文を生成します。")
#     parser.add_argument(
#         "--theme",
#         type=str,
#         required=True,
#         help="検索したいテーマを指定してください。",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="gpt-4o-2024-05-13",
#         choices=AVAILABLE_LLMS,
#         help="使用する LLM のモデルを指定してください。",
#     )
#     args = parser.parse_args()

#     # クライアントを作成
#     client, client_model = create_client(args.model)

#     # テーマに関連する論文を検索し、情報を収集
#     papers = collect_papers_on_theme(
#         theme=args.theme,
#     )

#     if papers:
#         # 論文情報を文字列にまとめる
#         papers_text = ""
#         for i, paper in enumerate(papers):
#             authors = ', '.join([author['name'] for author in paper['authors']])
#             papers_text += f"{i+1}. {paper['title']} by {authors} ({paper['year']})\n"
#             papers_text += f"Abstract: {paper['abstract']}\n\n"

#         # 論文情報をもとにレビュー論文を生成
#         generate_review_paper_from_papers(
#             papers_text=papers_text,
#             theme=args.theme,
#             client=client,
#             model=client_model,
#             output_file="generated_paper.tex"
#         )

#         # LaTeXファイルをコンパイルしてPDFを生成
#         compile_latex_file("generated_paper.tex", output_pdf="generated_paper.pdf")
#     else:
#         print("No papers were collected.")
