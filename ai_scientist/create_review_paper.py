import json
import os
import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional

import backoff
# 必要な外部パッケージをインポート
import requests
# LLM関連の関数を含むモジュールをインポート
from llm import AVAILABLE_LLMS, create_client, get_response_from_llm

# Semantic Scholar APIキーを環境変数から取得
S2_API_KEY = os.getenv("S2_API_KEY")

MAX_NUM_TOKENS = 8192  # モデルの上限トークン数に合わせて設定

def extract_text_between_markers(text: str, start_marker: str, end_marker: str) -> Optional[str]:
    pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return None

def generate_chapter_structure(theme: str, client, model) -> List[str]:
    prompt = f"""あなたは、テーマ「{theme}」に関する包括的なレビュー論文の章立てを作成するAIアシスタントです。

重要な側面をすべてカバーする章のタイトルのリストを提供してください。

サブチャプターや小チャプターは含めず、大きな章のみを含めてください。

出力は以下のJSON形式で提供してください：

<BEGIN OUTPUT>
{{
    "chapters": [
        "序論",
        "背景",
        ...
    ]
}}
<END OUTPUT>
"""

    response, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message="あなたは学術論文の章立てを生成する有能なアシスタントです。",
        msg_history=[],
        temperature=0.5  # 必要に応じて調整
    )

    # マーカー間のテキストを抽出
    json_text = extract_text_between_markers(response, "<BEGIN OUTPUT>", "<END OUTPUT>")
    if json_text is None:
        print("LLMの出力から章構成を抽出できませんでした。")
        return []

    try:
        chapters = json.loads(json_text).get("chapters", [])
    except json.JSONDecodeError:
        print("LLMの出力からJSONを解析できませんでした。")
        return []

    return chapters

def on_backoff(details):
    print(
        f"{time.strftime('%X')}に関数{details['target'].__name__}の呼び出しが{details['tries']}回失敗しました。{details['wait']:0.1f}秒待機します。"
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
    print(f"レスポンスステータスコード: {rsp.status_code}")
    print(
        f"レスポンス内容: {rsp.text[:500]}"
    )
    rsp.raise_for_status()
    results = rsp.json()
    total = results.get("total", 0)
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers

def summarize_papers_for_chapter(chapter_title: str, theme: str, client, model) -> str:
    # 章に関連する論文を検索
    query = f"{theme} {chapter_title}"
    papers = search_for_papers(query, result_limit=10)
    if not papers:
        print(f"章「{chapter_title}」に関連する論文が見つかりませんでした。")
        return ""
    
    # 論文情報を文字列にまとめる
    papers_text = ""
    for i, paper in enumerate(papers):
        authors = ', '.join([author['name'] for author in paper.get('authors', [])])
        papers_text += f"{i+1}. {paper.get('title', 'タイトルなし')}（{paper.get('year', '年不明')}）\n"
        papers_text += f"著者: {authors}\n"
        papers_text += f"要旨: {paper.get('abstract', '要旨なし')}\n\n"
    
    # LLMを使って論文を詳細にまとめる（日本語で）
    summary_prompt = f"""あなたは、提供された要旨に基づいて包括的な文献レビューを書くAIアシスタントです。

以下の論文の要旨をもとに、テーマ「{theme}」の章「{chapter_title}」に関連する詳細な内容を推測・展開してください。

各論文について、背景、目的、方法、結果、考察、結論を含む詳細な分析を提供してください。

また、これらの論文が研究の進展にどのように貢献し、互いにどのように関連しているかを総合的に述べてください。

さらに文章量を増やしたいため、あなたの知識を追加することを許可します。

以下が論文の要旨です。ただし英語の可能性があるため適宜日本語訳してください：

{papers_text}
"""

    summary, _ = get_response_from_llm(
        summary_prompt,
        client=client,
        model=model,
        system_message="あなたは知識豊富な研究者です。提供された要旨に基づいて包括的な文献レビューを書いてください。",
        msg_history=[],
        temperature=0.7  # 必要に応じて調整
    )

    return summary

def generate_latex_review_paper(theme, chapters_summaries, client, model, output_file="generated_review_paper.tex"):
    # プロンプトの作成（日本語で）
    prompt = f"""あなたは、提供された各章の要約に基づいて、包括的なレビュー論文をLaTeX形式で書くAIアシスタントです。

テーマ: 「{theme}」

日本語で、LaTeXクラスとして'jsarticle'を使用し、二段組みにしてください。必要なセクション（要旨、序論、背景、結論など）を含めてください。実験結果のセクションは除外してください。

提供された各章の要約を使って内容を構成してください。

LaTeXコードを<BEGIN LATEX>と<END LATEX>の間に提供してください。

以下が各章の要約です：

"""
    for chapter_title, summary in chapters_summaries.items():
        prompt += f"### {chapter_title}\n{summary}\n\n"

    # LLMを使用してLaTeXコードを生成
    response, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message="あなたは、'jsarticle'クラスを使用して二段組みの包括的なLaTeXレビュー論文を書く有能なアシスタントです。出力は日本語で提供してください。",
        msg_history=[],
        temperature=0.5  # 必要に応じて調整
    )

    # LaTeXコードの抽出
    latex_code = extract_text_between_markers(response, "<BEGIN LATEX>", "<END LATEX>")

    if latex_code is None:
        print("LLMの出力からLaTeXコードを抽出できませんでした。")
        return

    # LaTeXコードをファイルに書き込む
    with open(output_file, "w") as f:
        f.write(latex_code)

    print(f"LaTeXファイルが{output_file}に書き出されました。")

def compile_latex_file(tex_file, output_pdf="output.pdf", timeout=120):
    cwd = os.path.dirname(os.path.abspath(tex_file))
    tex_filename = os.path.basename(tex_file)
    commands = [
        ["uplatex", "-interaction=nonstopmode", tex_filename],
        ["uplatex", "-interaction=nonstopmode", tex_filename],
        ["dvipdfmx", tex_filename.replace(".tex", ".dvi")],
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
            print("標準出力:\n", result.stdout)
            print("標準エラー:\n", result.stderr)
        except subprocess.TimeoutExpired:
            print(f"{timeout}秒後にLaTeXのコンパイルがタイムアウトしました。")
            return
        except subprocess.CalledProcessError as e:
            print(f"コマンド{' '.join(command)}の実行中にエラーが発生しました: {e}")
            return

    # PDFファイルを出力
    generated_pdf = os.path.join(cwd, tex_filename.replace(".tex", ".pdf"))
    if os.path.exists(generated_pdf):
        shutil.move(generated_pdf, output_pdf)
        print(f"PDFファイルが{output_pdf}として生成されました。")
    else:
        print("PDFの生成に失敗しました。")

if __name__ == "__main__":
    # コマンドライン引数の解析
    import argparse

    parser = argparse.ArgumentParser(description="テーマに関連する論文を検索し、要約し、日本語で二段組みのレビュー論文を生成します。")
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

    theme = args.theme

    # LLMクライアントの作成
    client, client_model = create_client(args.model)

    # 1. 章構成の提案
    chapters = generate_chapter_structure(theme, client, client_model)

    if not chapters:
        print("章が生成されませんでした。")
        exit()

    # 2. 各章の処理
    chapters_summaries = {}
    for chapter_title in chapters:
        print(f"章を処理中: {chapter_title}")
        summary = summarize_papers_for_chapter(chapter_title, theme, client, client_model)
        chapters_summaries[chapter_title] = summary
        time.sleep(1)  # APIのレート制限を考慮して待機

    # 3. LaTeXレビュー論文の生成（二段組み、日本語）
    generate_latex_review_paper(
        theme=theme,
        chapters_summaries=chapters_summaries,
        client=client,
        model=client_model,
        output_file="generated_review_paper.tex"
    )

    # 4. LaTeXファイルのコンパイル
    compile_latex_file("generated_review_paper.tex", output_pdf="generated_review_paper.pdf")
