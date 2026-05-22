"""
summarize.py — combine every JSON in ../results/ into one summary CSV/markdown.

Run AFTER experiments finish (`make_figs.py` produces the pictures, this
produces the table).
"""

import os, sys, json, glob

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_all():
    rows = []
    for path in sorted(glob.glob(os.path.join(RES, '*.json'))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            h = json.load(f)
        row = {
            'name': name,
            'best_dev_acc': h.get('best_dev_acc', 0.0),
            'test_acc': h.get('test_acc', 0.0),
            'epochs': len(h.get('epoch_dev_acc', [])),
        }
        for k in ('lr', 'lambda', 'dropout', 'tag'):
            if k in h:
                row[k] = h[k]
        rows.append(row)
    return rows


def main():
    rows = load_all()
    # CSV
    out_csv = os.path.join(RES, 'summary.csv')
    cols = ['name', 'epochs', 'best_dev_acc', 'test_acc', 'lr', 'lambda', 'dropout', 'tag']
    with open(out_csv, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r.get(c, '')) for c in cols) + '\n')
    print(f'wrote {out_csv}')

    # Markdown
    out_md = os.path.join(RES, 'summary.md')
    with open(out_md, 'w') as f:
        f.write('| Run | Epochs | Best val | Test acc |\n|---|---|---|---|\n')
        for r in sorted(rows, key=lambda x: -x['test_acc']):
            f.write(f'| `{r["name"]}` | {r["epochs"]} | '
                    f'{r["best_dev_acc"]*100:.2f}% | {r["test_acc"]*100:.2f}% |\n')
    print(f'wrote {out_md}')

    print('\nTop 10 by test acc:')
    for r in sorted(rows, key=lambda x: -x['test_acc'])[:10]:
        print(f'  {r["name"]:<40} dev={r["best_dev_acc"]:.4f}  test={r["test_acc"]:.4f}')


if __name__ == '__main__':
    main()
