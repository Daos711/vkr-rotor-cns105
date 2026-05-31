import re, glob, os

root = '/home/user/vkr-rotor-cns105'
# content files only (skip preamble/main where --- is decorative in comments,
# and skip this script)
files = ['bibliography.tex',
         'sections/abstract.tex',
         'sections/task.tex',
         'sections/ch1.tex',
         'sections/chapters_2_3.tex',
         'sections/introduction.tex',
         'sections/conclusion.tex']

report = []
for rel in files:
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        continue
    s = open(p, encoding='utf-8').read()
    orig = s
    # 1) number ranges: digit--digit -> digit-digit (hyphen)
    n_range = len(re.findall(r'(?<=\d)--(?=\d)', s))
    s = re.sub(r'(?<=\d)--(?=\d)', '-', s)
    # 2) em-dash (---) -> en-dash (--)  [only real content; these files
    #    have no decorative --- comment rules]
    n_em = s.count('---')
    s = s.replace('---', '--')
    if s != orig:
        open(p, 'w', encoding='utf-8').write(s)
    report.append(f'{rel}: ranges->hyphen={n_range}, em->en={n_em}')

open(os.path.join(root, 'fix_dashes_report.txt'), 'w', encoding='utf-8').write('\n'.join(report))
