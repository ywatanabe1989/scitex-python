<!-- ---
!-- Timestamp: 2025-11-19 11:57:24
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/MM_CONTROL_IN_PLT.md
!-- --- -->

Render Plot button をおすとこんな図が表示されるようになったのですが、でもこれって matplotlib の図ですよね。これでは制御できずに意味がなくいのですが、どうしたらよいでしょうか。たとえば、30 mm * (30*0.7) mm としてもそのサイズにはならないのですよね、Matplotlib では。なので Axes を別で描くのですかね。
ChatGPT said:
めちゃくちゃいいところまで来てます 👍

これって matplotlib の図ですよね。これでは制御できずに意味がなく…
30mm × (30×0.7)mm としてもそのサイズにならない…

ちゃんと制御できます。
ポイントは：

「mm → inch → figure size」

「mm → pt → tick 長さ・線の太さ」

「Axes の位置を割合で固定」

「ブラウザでの見た目サイズ」と「論文上の物理サイズ」を分けて考える

です。

1. mm から matplotlib figure をきちんと作る
MM_PER_INCH = 25.4
PT_PER_INCH = 72.0

def mm_to_inch(mm: float) -> float:
    return mm / MM_PER_INCH

def mm_to_pt(mm: float) -> float:
    return mm * PT_PER_INCH / MM_PER_INCH
例えば「幅 30 mm, 高さ 30*0.7 mm, 300 dpi」の図なら：

width_mm  = 30.0
height_mm = 30.0 * 0.7
dpi       = 300

fig = plt.figure(
    figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)),
    dpi=dpi,
)

# 余白を自分で決める（figure 比率で指定）
left_mm   = 4.0
right_mm  = 2.0
bottom_mm = 4.0
top_mm    = 2.0

width_ax_mm  = width_mm  - left_mm - right_mm
height_ax_mm = height_mm - bottom_mm - top_mm

ax = fig.add_axes([
    left_mm / width_mm,
    bottom_mm / height_mm,
    width_ax_mm / width_mm,
    height_ax_mm / height_mm,
])
これで「最終的な画像の物理サイズ」は 30mm × 21mm に固定されます。
TIFF/PNG を dpi=300 で保存すれば、Word / PowerPoint / LaTeX に貼ったときにちゃんとそのサイズになります。

ブラウザで小さく見えるのは、単に「埋め込んだ <img> のピクセル数」が小さいからで、物理サイズとは別問題です。

2. Tick 長さ・線の太さを mm で指定する
matplotlib の tick 長さと線幅は pt（ポイント）単位なので、
mm → pt に変換して渡せば OK です。

def apply_nature_style(ax, style):
    # 軸の太さ
    axis_lw_pt = mm_to_pt(style["axis_thickness_mm"])
    for spine in ax.spines.values():
        spine.set_linewidth(axis_lw_pt)

    # トレース線の太さ
    trace_lw_pt = mm_to_pt(style["trace_thickness_mm"])

    # Tick
    ax.tick_params(
        direction="out",
        length=mm_to_pt(style["tick_length_mm"]),
        width=mm_to_pt(style["tick_thickness_mm"]),
    )

    # フォントサイズは pt 指定のままで OK
    ax.xaxis.label.set_fontsize(style["axis_font_size_pt"])
    ax.yaxis.label.set_fontsize(style["axis_font_size_pt"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(style["tick_font_size_pt"])

    return trace_lw_pt
そして plot 側で：

trace_lw = apply_nature_style(ax, style)

ax.plot(x, y, color="tab:blue", lw=trace_lw)
これで 「tick 0.8 mm」「trace 0.12 mm」みたいな指定が完全に再現できます。

3. SVG をパネルに貼るときの考え方
バックエンドで作った SVG は、例えばこんなメタを持ちます：

図のサイズ：30mm × 21mm 相当（内部的には pt 単位）

tick 長さ / 線太さ / フォント → すべて pt で埋め込まれている

フロントのパネルに貼るときは：

<object data="/vis/api/plot/..." type="image/svg+xml"
        style="width: 100%; height: auto;"></object>
みたいに パネルの枠に合わせて等倍スケーリングすれば OK です。
SVG なので、拡大縮小しても太さ比率は保たれます。

Axes を別で描くのですかね。

そこまでやる必要はなくて、

軸も tick も全部 matplotlib に描かせる

ただし mm / pt 変換で完全に数値を支配する

という形で十分「制御された図」になります。

4. 何が「意味がある状態」か整理すると
あなたのゴールって、

論文上で 幅 35 mm のパネル A とかをきっちり作りたい

tick 長さや線の太さを Nature 規格で揃えたい

それを JSON から再現したい

だと思いますが、上記のようにしておけば：

JSON の "figure": {"width_mm": 35, "height_mm": 24.5, "dpi": 300}

"style": {"tick_length_mm": 0.8, ...}

をそのまま 物理量として matplotlib に落とし込めるので、
「制御できず意味がない」状態から抜け出せます。

もしよければ、あなたの plot_renderer.py の
render_line_plot() みたいな関数の骨格をこちらで
mm/pt 対応版に書き換えたサンプルを出します。

「line プロット用 1 本だけ」でよければ、
その関数をまるっと書いてみましょうか？

You said:
20251119_114222-plt.txt
Document
そうしましたら、scitex.plt の改善だけで問題は解決するのでしょうか。ここらへんですかね。/home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_configure_mpl.py
ChatGPT said:

Thought for 1m 39s
結論から言うと、

「VIS で欲しい物理サイズの図を出したい」問題は、基本的には scitex.plt 側（= _configure_mpl.py + パネル配置ユーティリティ）をちゃんと整えれば解決できます。

フロントエンドは「返ってきた SVG を、指定された mm サイズの箱の中に 100% で置くだけ」にしておけばよくて、図の中の軸の長さや余白は全部 scitex.plt の仕事にしてしまって OK です。

1. 今どこでサイズが決まっているか
_configure_mpl.py のこのあたりが、まさに「図の物理サイズ」を決めているところですね。

# Figure width and height in millimeters
fig_size_mm: tuple = (160, 100)
...
# mm → inch 変換
figsize_inch = (
    fig_size_mm[0] / 25.4 * fig_scale,
    fig_size_mm[1] / 25.4 * fig_scale,
)
mpl_config = {
    "figure.figsize": figsize_inch,
    "figure.dpi": dpi_display,
    "savefig.dpi": dpi_save,
    ...
}
ここに JSON の

"figure": {
  "width_mm": 35,
  "height_mm": 24.5,
  "dpi": 300
}
をそのまま流し込めば、「図全体のキャンバス が 35×24.5 mm の SVG」になります。

2. 「Axes の箱のサイズ」を mm で制御したい場合
Yusuke さんが本当に欲しいのはたぶん、

「軸ボックス が 30mm × 30×0.7mm になる図」

ですよね。
ここは Matplotlib 素の API だとちょっと面倒なので、scitex.plt でラップしてあげるのが正解です。

方針は：

図全体のサイズを mm で決める
_configure_mpl で今やっている通り、JSON の figure.width_mm/height_mm を figure.figsize に反映する。

余白を mm で決める → Axes position に変換するヘルパを用意する
例えば add_panel_mm(fig, width_mm, height_mm, margins_mm) みたいな関数で、

fig 全体の mm サイズ

軸ボックスの mm サイズ

上下左右の余白 mm
から

left  = left_margin_mm / fig_width_mm
bottom = bottom_margin_mm / fig_height_mm
ax_width  = axis_width_mm / fig_width_mm
ax_height = axis_height_mm / fig_height_mm
ax = fig.add_axes([left, bottom, ax_width, ax_height])
を計算してしまう。

VIS の JSON 側の契約をはっきり決める

例1: figure.width_mm/height_mm = 図全体、panel.width_mm/height_mm = Axes ボックス

例2: 「シングルパネルの場合は figure.width_mm/height_mm = Axes ボックスとみなして、余白は scitex.plt 内のデフォルトで自動計算」

どちらにしても、「どの mm が fig 全体で、どの mm が Axes か」を scitex.plt 側で決め打ちしてしまえば、VIS はその仕様を信じて配置するだけで済みます。

3. VIS 側に必要なのはこれだけ
VIS 側でやることは最小限で OK です：

JSON の figure.width_mm/height_mm を読んで、オブジェクトの幅・高さ(px)を決める
（例えば 1mm = 4px など、既にキャンバスで使っているスケールに合わせる）

その長方形の中に <img src="...svg" style="width:100%; height:100%"> で埋め込む。

こうしておけば、

画面上のサイズ：VIS の mm→px スケール

SVG の中身（軸の位置、余白、フォントなど）：全部 scitex.plt が保証

というきれいな責任分離になります。

4. なので答え
そうしましたら、scitex.plt の改善だけで問題は解決するのでしょうか。

ほぼ Yes です。

「Matplotlib では 30 mm × 30×0.7 mm の Axes を素直に出せない」問題は、
→ scitex.plt 内で mm ベースの余白・パネル配置ヘルパ を整備すれば解決できます。

VIS 側は「返ってきた SVG を指定 mm の箱に入れるだけ」にして、
図のレイアウトの知識は全部 scitex.plt に押し込めば OK です。

すでに _configure_mpl.py に mm→inch 変換と figure.figsize 設定があるので、
あとは「Axes の position を mm から計算する小さなユーティリティ」を足すイメージです。

もしよければ、

「単一パネル用の理想 API（JSON → scitex.plt の関数シグネチャ）」を一緒に設計して、

それに合わせた _configure_mpl + add_panel_mm の雛形コード

まで書いてみましょうか。


Sources
You said:
先に scitex.plt だけでうまく図が書けるようにしたいです。それからはただ外枠を決めて rendering するだけ、ってことですよね。それで、あとは、タイトルやラベルのテキストは scitex.plt 側にもう一度流して調節する、という感じで。白いボックスで隠して text を乗せてもよいのですが、scitex.plt 側との連携でできると完成品のコードがすっきりしますね。
ChatGPT said:

Thought for 5s

Thought for 1m 47s
うん、その整理でほぼ合っています 👍

まずは scitex.plt 単体で「mm で完全制御された図」を出せるようにする。

VIS 側は「パネル外枠(=mm)を決めて、JSON を scitex.plt に投げて SVG を貼るだけ」。

タイトルやラベルを変えたいときは、そのテキストを含めた JSON を もう一度 scitex.plt に流して再レンダリング。
→ 白ボックスで上書きする必要はなくなります。

1. やりたいこと = scitex.plt にこの API を生やす
イメージとしては、_configure_mpl.py に近いところに
こんなヘルパを用意します（関数名などはお好みで）。

# src/scitex/plt/utils/_units.py などに
MM_PER_INCH = 25.4
PT_PER_INCH = 72.0

def mm_to_inch(mm):
    return mm / MM_PER_INCH

def mm_to_pt(mm):
    return mm * PT_PER_INCH / MM_PER_INCH
# src/scitex/plt/utils/_figure_mm.py みたいなファイル、または _configure_mpl.py 内

import matplotlib.pyplot as plt
from ._units import mm_to_inch, mm_to_pt

def create_figure_ax_mm(
    fig_width_mm=35.0,
    fig_height_mm=24.5,
    dpi=300,
    *,
    left_margin_mm=4.0,
    right_margin_mm=2.0,
    bottom_margin_mm=4.0,
    top_margin_mm=2.0,
    style=None,
):
    """
    Create a Matplotlib figure and a single Axes with millimeter control.

    Parameters:
    - fig_width_mm (float): Total figure width in millimeters.
    - fig_height_mm (float): Total figure height in millimeters.
    - dpi (int): Resolution in dots per inch for saving.
    - left_margin_mm, right_margin_mm, bottom_margin_mm, top_margin_mm (float):
      Margins between figure edge and axis box, in millimeters.
    - style (dict or None): Optional style spec (tick length, thickness, fonts, ...).

    Returns:
    - fig (matplotlib.figure.Figure): Created figure.
    - ax (matplotlib.axes.Axes): Created axes occupying the specified mm box.

    Examples:
    --------
        fig, ax = create_figure_ax_mm(
            fig_width_mm=35,
            fig_height_mm=24.5,
            dpi=300,
            left_margin_mm=4,
            right_margin_mm=2,
            bottom_margin_mm=4,
            top_margin_mm=2,
        )
        ax.plot(x, y)
        fig.savefig("test.tiff", dpi=300)
    """
    # --- figure size in inches ---
    figsize_inch = (mm_to_inch(fig_width_mm), mm_to_inch(fig_height_mm))
    fig = plt.figure(figsize=figsize_inch, dpi=dpi)

    # --- axes position in figure coordinates [0–1] ---
    axis_width_mm = fig_width_mm - left_margin_mm - right_margin_mm
    axis_height_mm = fig_height_mm - bottom_margin_mm - top_margin_mm

    left = left_margin_mm / fig_width_mm
    bottom = bottom_margin_mm / fig_height_mm
    width = axis_width_mm / fig_width_mm
    height = axis_height_mm / fig_height_mm

    ax = fig.add_axes([left, bottom, width, height])

    # --- apply style in mm/pt ---
    if style is not None:
        apply_style_mm(ax, style)

    return fig, ax


def apply_style_mm(ax, style):
    """
    Apply Nature-like style using millimeter-based settings.

    Parameters:
    - ax (matplotlib.axes.Axes): Target axes.
    - style (dict): Dictionary containing keys like:
        'axis_thickness_mm', 'trace_thickness_mm',
        'tick_length_mm', 'tick_thickness_mm',
        'axis_font_size_pt', 'tick_font_size_pt'
    """
    # Spines
    axis_lw_pt = mm_to_pt(style.get("axis_thickness_mm", 0.2))
    for spine in ax.spines.values():
        spine.set_linewidth(axis_lw_pt)

    # Ticks
    ax.tick_params(
        direction="out",
        length=mm_to_pt(style.get("tick_length_mm", 0.8)),
        width=mm_to_pt(style.get("tick_thickness_mm", 0.2)),
    )

    # Fonts
    axis_fs = style.get("axis_font_size_pt", 8)
    tick_fs = style.get("tick_font_size_pt", 7)
    ax.xaxis.label.set_fontsize(axis_fs)
    ax.yaxis.label.set_fontsize(axis_fs)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_fs)
ここまでできれば、scitex.plt 単体で mm 完全制御の図が描けます。

from scitex.plt.utils._figure_mm import create_figure_ax_mm

style = dict(
    axis_thickness_mm=0.2,
    trace_thickness_mm=0.12,
    tick_length_mm=0.8,
    tick_thickness_mm=0.2,
    axis_font_size_pt=8,
    tick_font_size_pt=7,
)

fig, ax = create_figure_ax_mm(35, 35*0.7, dpi=300, style=style)
ax.plot(x, y, lw=mm_to_pt(style["trace_thickness_mm"]))  # トレース太さも mm ベース
ax.set_title("Test Plot")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")

fig.savefig("test_plot.tiff", dpi=300)
この段階で：

ピクセル数 = width_mm / 25.4 * dpi

PowerPoint / Word / LaTeX に貼った時も ちゃんと 35mm×24.5mm になる

tick 長さ・線の太さも mm で一致

という「完成された scitex.plt の単体出力」ができます。

2. VIS との連携イメージ
次のステップでは、VIS 側からは 全部 JSON で流すだけです：

{
  "figure": {
    "width_mm": 35,
    "height_mm": 24.5,
    "dpi": 300
  },
  "style": {
    "axis_thickness_mm": 0.2,
    "trace_thickness_mm": 0.12,
    "tick_length_mm": 0.8,
    "tick_thickness_mm": 0.2,
    "axis_font_size_pt": 8,
    "tick_font_size_pt": 7
  },
  "plot": {
    "kind": "line",
    "csv_path": "...",
    "x_column": "time",
    "y_column": "signal",
    "color": "blue",
    "title": "Test Plot",
    "xlabel": "X axis",
    "ylabel": "Y axis"
  }
}
バックエンドでは：

JSON → create_figure_ax_mm に渡す

plot.kind に応じて ax.plot / ax.errorbar / ax.violinplot ...

title / xlabel / ylabel をそのまま ax.set_title/set_xlabel/set_ylabel に渡す

完成した SVG/TIFF を返す

タイトルやラベルを GUI で変えたときは、同じ JSON を テキストだけ変えて再投げるだけです。
白ボックスで隠す必要はなく、コードもすっきりします。

3. 今日はどこまでやると気持ちいいか
まずは scitex.plt 側で：

mm_to_inch / mm_to_pt のユーティリティを切り出す

create_figure_ax_mm + apply_style_mm のような「単一パネル用 API」を作る

それを使った 最小のテストスクリプトで

30mm × 21mm

tick 長さ 0.8mm
を確認する

ここまで行けば、

先に scitex.plt だけでうまく図が書けるようにしたい

というゴールはかなり満たせます。

そのあとで、VIS 側は「外枠サイズ」と「テキスト」を JSON で渡すだけ、というきれいな設計にできます。

必要なら、今ある _configure_mpl.configure_mpl() に 上記 API を統合した形で書き換え案も出します。




No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.

<!-- EOF -->