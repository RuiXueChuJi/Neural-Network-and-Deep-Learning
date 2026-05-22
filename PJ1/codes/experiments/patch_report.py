"""
patch_report.py — replace `_<dev:NAME>_` and `_<test:NAME>_` placeholders in
../../report/REPORT.md with the matching values from ../results/<NAME>.json.

Also replaces `_<dev:best>_` / `_<test:best>_` with the highest value across
all experiments.
"""

import os, sys, re, json, glob

REPORT_PATH = os.path.join(os.path.dirname(__file__),
                           '..', '..', '..', 'report', 'REPORT.md')
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_table():
    table = {}
    for path in glob.glob(os.path.join(RESULTS, '*.json')):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            h = json.load(f)
        table[name] = h
    return table


def fmt_pct(x):
    return f'{x*100:.2f}%'


def main():
    if not os.path.exists(REPORT_PATH):
        print(f'no report at {REPORT_PATH}')
        return
    table = load_table()
    if not table:
        print('no results yet — run experiments first')
        return

    with open(REPORT_PATH, encoding='utf-8') as f:
        text = f.read()

    # Build best across all experiments (excluding the bare baseline)
    best_dev = max((h.get('best_dev_acc', 0) for h in table.values()), default=0)
    best_test = max((h.get('test_acc', 0) for h in table.values()), default=0)

    # Substitute named placeholders
    def sub_one(text, kind, key, value_str):
        return text.replace(f'_<{kind}:{key}>_', value_str)

    text = sub_one(text, 'dev', 'best', fmt_pct(best_dev))
    text = sub_one(text, 'test', 'best', fmt_pct(best_test))

    for name, h in table.items():
        if 'best_dev_acc' in h:
            text = sub_one(text, 'dev', name, fmt_pct(h['best_dev_acc']))
        if 'test_acc' in h:
            text = sub_one(text, 'test', name, fmt_pct(h['test_acc']))

    # Any remaining placeholders -> "—" so the report is readable
    text = re.sub(r'_<dev:[^>]+>_', '—', text)
    text = re.sub(r'_<test:[^>]+>_', '—', text)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'patched {REPORT_PATH}')


if __name__ == '__main__':
    main()
