#!/usr/bin/env bash
# 生成技术报告 PDF。依赖：pandoc、xelatex（TeX Live）、Noto CJK 字体（Noto Serif/Sans/Sans Mono CJK SC）。
# 用法：bash docs/report/build.sh
# 可复现：时间戳取最后一次提交时间（SOURCE_DATE_EPOCH），时区固定为 UTC，PDF 元数据不含本机时区。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct 2>/dev/null || date +%s)"
export SOURCE_DATE_EPOCH
export TZ=UTC
pandoc report.md -o report.pdf --pdf-engine=xelatex --resource-path=. \
  -V mainfont="Noto Serif CJK SC" -V sansfont="Noto Sans CJK SC" -V monofont="Noto Sans Mono CJK SC" \
  -V CJKmainfont="Noto Serif CJK SC" -V CJKsansfont="Noto Sans CJK SC" -V CJKmonofont="Noto Sans Mono CJK SC" \
  -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true -M toc-title="目录" --toc
echo "report.pdf: $(du -h report.pdf | cut -f1)"
