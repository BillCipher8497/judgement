"""Render the validated Markdown card to one A4 page; preserve all wording."""
import hashlib,html,json,re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent;ENGINE=HERE.parents[2]
SOURCE=ENGINE/'paper/guide_card.md';OUT=HERE/'results';OUT.mkdir(exist_ok=True)
fonts=next(p for p in [Path('C:/Windows/Fonts'),Path('/usr/share/fonts/truetype/dejavu')]
           if (p/'arial.ttf').exists() or (p/'DejaVuSans.ttf').exists())
regular=fonts/('arial.ttf' if (fonts/'arial.ttf').exists() else 'DejaVuSans.ttf')
bold=fonts/('arialbd.ttf' if (fonts/'arialbd.ttf').exists() else 'DejaVuSans-Bold.ttf')
pdfmetrics.registerFont(TTFont('Card',str(regular)));pdfmetrics.registerFont(TTFont('CardBold',str(bold)))
pdfmetrics.registerFontFamily('Card',normal='Card',bold='CardBold',italic='Card',boldItalic='CardBold')
styles={'body':ParagraphStyle('body',fontName='Card',fontSize=9.1,leading=11.6,spaceAfter=6),
        'title':ParagraphStyle('title',fontName='CardBold',fontSize=18,leading=21,spaceAfter=9,textColor=colors.HexColor('#142e43')),
        'section':ParagraphStyle('section',fontName='CardBold',fontSize=11.3,leading=14,spaceBefore=6,spaceAfter=7,textColor=colors.HexColor('#142e43')),
        'table':ParagraphStyle('table',fontName='Card',fontSize=9.1,leading=11.6),
        'rule':ParagraphStyle('rule',fontName='Card',fontSize=9.1,leading=11.6,spaceAfter=6,leftIndent=13,firstLineIndent=-13)}
def normalize(s):return s.replace('—','-').replace('−','-').replace('–','-')
def rich(s):
    s=html.escape(normalize(s).replace('`',''))
    s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    return s.strip('*')
lines=SOURCE.read_text(encoding='utf-8').splitlines();story=[];expected=[];i=0
while i<len(lines):
    line=lines[i].strip();i+=1
    if not line:continue
    if line.startswith('|'):
        table=[]
        while line.startswith('|'):
            cells=[c.strip() for c in line.strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c) for c in cells):
                table.append([Paragraph(rich(c),styles['table']) for c in cells]);expected.extend(cells)
            if i>=len(lines):break
            line=lines[i].strip()
            if not line.startswith('|'):break
            i+=1
        t=Table(table,colWidths=[(A4[0]-72)/2]*2,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eff3')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
            ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#8195a3'))]))
        story.extend([t,Spacer(1,8)]);continue
    if line.startswith('# '):text=line[2:];kind='title'
    elif line.startswith('## '):text=line[3:];kind='section'
    else:text=line;kind='rule' if re.match(r'^\d+\. ',line) else 'body'
    expected.append(text);story.append(Paragraph(rich(text),styles[kind]))
dest=OUT/'guide_card.pdf'
doc=SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=36,leftMargin=36,topMargin=32,bottomMargin=30,
                      title='Judgement - table card',author='Judgement Research')
doc.build(story)
reader=PdfReader(dest);assert len(reader.pages)==1,'Card must render as exactly one page'
actual='\n'.join(p.extract_text() for p in reader.pages)
tokens=lambda s:re.findall(r'\w+',normalize(s).replace('`','').replace('*',''))
assert tokens(' '.join(expected))==tokens(actual),'PDF wording differs from Markdown'
audit={'source':'paper/guide_card.md','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
       'pdf_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'pages':len(reader.pages),
       'page_size':'A4','word_tokens':len(tokens(actual)),'all_source_words_preserved_in_order':True,
       'font_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [regular,bold]},
       'new_strategy_or_empirical_results':False,'visual_QA':'pending rendered PNG inspection'}
(OUT/'render_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
